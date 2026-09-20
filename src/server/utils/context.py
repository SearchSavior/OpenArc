"""Resolve a model's context-window length.

OpenArc uses the resolved context window for two things that must stay in
lockstep:

  1. *Advertisement* -- it is reported in the ``GET /v1/models`` response
     (see :mod:`src.server.routes.openai`, under both the OpenAI-standard
     ``context_window`` field and ``meta.n_ctx`` for goose) so that clients can
     size their conversation and, e.g. trigger auto-compaction before the window
     is exceeded.

  2. *Enforcement* -- it is fed to the inference engine (see
     :func:`src.server.model_registry.ModelRegistry.register_load` and
     :mod:`src.engine.ov_genai.utils`) and becomes the compiled pipeline's
     **max content window** (openvino.genai's ``SchedulerConfig.max_num_batched_tokens``),
     which bounds a running sequence's KV-cache growth. So an explicit
     ``--context-window`` or a ``config.json`` value actually caps inference,
     not merely what /v1/models shows.

The canonical value is discovered from the model's own ``config.json``
(shipped next to the OpenVINO IR in ``model_path``). Different model exports
name the overall context window differently, so we honour the first key in
:func:`CONTEXT_FIELD_PRIORITY` that is actually present in ``config.json``:

- ``max_position_embeddings`` : the HuggingFace standard key
- ``n_positions`` / ``seq_len`` / ``seq_length`` : llama.cpp / legacy HF forms
- ``n_ctx``                    : llama.cpp server / GGUF field
- ``sliding_window``           : last resort (may be far smaller than the
                                  true window for hybrid / sparse-attention
                                  models), hence checked last
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Keys that may hold a model's overall context length. The first key that is
# present (with a positive integer value) wins, in this order.
CONTEXT_FIELD_PRIORITY: List[str] = [
    "max_position_embeddings",
    "n_positions",
    "seq_len",
    "seq_length",
    "n_ctx",
    "sliding_window",
]


def _coerce_positive_int(value: object) -> Optional[int]:
    """Return ``value`` as a positive ``int`, else ``None``.

    ``None``, booleans, non-numbers, and non-positive numbers do not count
    as a usable window size, so they are rejected (``None`` is returned and
    the caller should try the next candidate key).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    return None


def read_context_window_from_config(model_path: str) -> Optional[int]:
    """Return the context-window length declared in ``model_path/config.json``.

    Walks :data:`CONTEXT_FIELD_PRIORITY` and returns the value of the first
    key that is present *and holds a positive integer*. Returns ``None`` when
    ``config.json`` is missing, unreadable, malformed, or contains no usable
    key. This function never raises.
    """
    try:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("context discovery: cannot read %s (%s)", config_path, exc)
            return None

        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug(
                "context discovery: malformed config.json at %s (%s)", config_path, exc
            )
            return None
        if not isinstance(config, dict):
            return None

        for key in CONTEXT_FIELD_PRIORITY:
            candidate = _coerce_positive_int(config.get(key))
            if candidate is not None:
                return candidate

        logger.debug(
            "context discovery: no context key in %s (looked for %s)",
            config_path,
            ", ".join(CONTEXT_FIELD_PRIORITY),
        )
        return None
    except Exception as exc:  # defensive: discovery must never break model loading
        logger.debug("context discovery failed for %s (%s)", model_path, exc)
        return None


def resolve_context_window(model_path: str, explicit: Optional[int] = None) -> Optional[int]:
    """Resolve the effective context-window length for a model.

    An explicit positive ``context_window`` (taken from the load config, e.g.
    via ``openarc add --context-window`` or ``/openarc/load``) takes precedence
    and is returned as-is. Otherwise the value is derived from the model's
    ``config.json`` using :func:`read_context_window_from_config`.

    The returned value is used for both *advertising* in ``/v1/models`` and as
    the compiled pipeline's **max content window** (see the module docstring).
    """
    if explicit is not None:
        resolved = _coerce_positive_int(explicit)
        if resolved is not None:
            return resolved
    return read_context_window_from_config(model_path)
