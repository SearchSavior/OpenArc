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

The canonical value is *discovered* from the model's own ``config.json``
(shipped next to the OpenVINO IR in ``model_path``). Different model exports
name the overall context window differently, so for each candidate key in
:data:`CONTEXT_FIELD_PRIORITY` (first present key wins) we look at the top level
of ``config.json`` **and**, when that key is not there, descend into the
per-modality sub-configs::

    config.json
    +- (flat top-level keys : plain LLMs / GGUFs / legacy HF)
    +- a nested section  (text_config / language_config / llm_config / ...)
       +- max_position_embeddings / n_positions / ...   (the real window)
    +- other nested sections (vision_config / audio_config / ...)

This nesting matters: multimodal (``*ForConditionalGeneration``) models --
Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5 / Gemma3 / Mistral-Small, ... -- do
**not** put ``max_position_embeddings`` at the top level of ``config.json``; it
is nested inside ``text_config`` (or ``language_config``). A flat top-level-only
scan therefore finds *nothing* and leaves the window both *un-advertised*
(absent from ``/v1/models``) and *un-enforced* (no ``max_num_batched_tokens``)
for exactly those models -- which is the bug this module's section scan fixes.
Flat configs (plain LLMs) still resolve from the top level exactly as before.

Candidate keys, highest priority first:

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

# When a config.json nests its per-modality configs (typical of multimodal /
# "*ForConditionalGeneration" models), the real window lives inside one of
# these top-level sections rather than at the top level. They are searched, in
# this order, before we fall back to a generic scan of any nested dictionary,
# so a model's *language* config wins over, say, a vision/audio config that could
# also contain a `sliding_window`-style key.
CONTEXT_SECTION_PRIORITY: List[str] = [
    "text_config",
    "language_config",
    "language_model",
    "llm_config",
    "text_model",
]

# Depth cap for the recursive section scan. Real model configs are at most a
# level or two deep, so a small bound avoids both a stack blow-up and ever
# wandering into some unrelated, deeply-nested subtree of a config.json.
_CONTEXT_SEARCH_DEPTH: int = 6


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


def _find_context_key(
    config: object, key: str, depth: int = _CONTEXT_SEARCH_DEPTH
) -> Optional[int]:
    """Return the first *positive* value stored at ``key`` anywhere in ``config``.

    The current (dictionary) level is consulted first, so a flat top-level value
    still takes precedence over a nested one. When the key is absent here we
    descend into the nested per-modality sub-configs -- named language/text
    sections (see :data:`CONTEXT_SECTION_PRIORITY`) before any other nested
    dictionary -- so a model that nests e.g. ``max_position_embeddings`` inside
    ``text_config`` is still discovered. Returns ``None`` when no usable value is
    found at or below this level.
    """
    if not isinstance(config, dict):
        return None

    in_scope = _coerce_positive_int(config.get(key))
    if in_scope is not None:
        return in_scope

    if depth <= 0:
        return None

    # 1) named language/text sections first (the model's real context lives there
    #    for multimodal configs); they win over any other nested dictionary.
    for section_name in CONTEXT_SECTION_PRIORITY:
        if isinstance(config.get(section_name), dict):
            hit = _find_context_key(config[section_name], key, depth - 1)
            if hit is not None:
                return hit

    # 2) fallback: any other (un-named) nested dictionary, in file order.
    handled = set(CONTEXT_SECTION_PRIORITY)
    for name, value in config.items():
        if name not in handled and isinstance(value, dict):
            hit = _find_context_key(value, key, depth - 1)
            if hit is not None:
                return hit

    return None


def read_context_window_from_config(model_path: str) -> Optional[int]:
    """Return the context-window length declared in ``model_path/config.json``.

    Walks :data:`CONTEXT_FIELD_PRIORITY` and returns the value of the first key
    holding a positive integer. Each key is looked up at the top level of
    ``config.json`` first, then -- for multimodal models that nest their
    per-modality configs -- inside the nested sections of
    :data:`CONTEXT_SECTION_PRIORITY` (see :func:`_find_context_key`). Returns
    ``None`` when ``config.json`` is missing, unreadable, malformed, or holds no
    usable key at the top level or in any nested section. This function never
    raises.
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

        # Top level first (flat LLMs / GGUFs / legacy HF); fall through to the
        # nested per-modality sub-configs when a key is not at the top level
        # (multimodal models -- Qwen*VL / Qwen3.5 / Gemma3 / ... -- nest it under
        # ``text_config``). A nested higher-priority key therefore still wins over
        # a top-level lower-priority one, matching the "first present wins" rule.
        for key in CONTEXT_FIELD_PRIORITY:
            candidate = _find_context_key(config, key)
            if candidate is not None:
                return candidate

        logger.debug(
            "context discovery: no context key in %s (looked for %s at the top level and in "
            "nested sections %s)",
            config_path,
            ", ".join(CONTEXT_FIELD_PRIORITY),
            ", ".join(CONTEXT_SECTION_PRIORITY),
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
