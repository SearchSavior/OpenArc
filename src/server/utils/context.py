"""Resolve an (optional) model context window for advertisement in /v1/models.

Advertisement is opt-in via ``load_config.context_window``. Accepted values:
- unset (``None``): nothing is read, nothing is advertised.
- ``"auto"``: read the model's ``max_position_embeddings`` from ``config.json``.
- a positive integer: advertised as-is.

Only ``max_position_embeddings`` is read from ``config.json`` -- other names,
which some exporters use for the same concept are not consulted. Multimodal
models nest it under their per-modality sections (text_config / etc.), so the
search also descends into those named sections.

A pinned integer larger than the model's real ``max_position_embeddings`` is
flagged by :func:`check_context_window_exceeded`; the load path warns loudly.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# The single key read from config.json; the one canonical name for the window.
CONTEXT_FIELD = "max_position_embeddings"

# Multimodal ("*ForConditionalGeneration") models nest max_position_embeddings
# under one of these top-level sections instead of at the top level.
CONTEXT_SECTION_PRIORITY: list[str] = [
    "text_config",
    "language_config",
    "language_model",
    "llm_config",
    "text_model",
]

# Cap the recursive section scan (real configs are at most a level or two deep).
_CONTEXT_SEARCH_DEPTH: int = 6


def _coerce_positive_int(value: object) -> Optional[int]:
    """Coerce a value to a positive int, else None (incl. the "auto" string)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed: Optional[float] = int(text)
        except ValueError:
            try:
                parsed = float(text)
            except ValueError:
                return None
        return int(parsed) if parsed > 0 else None
    return None


def is_auto(value: object) -> bool:
    """True for the special ``"auto"`` value (case-insensitive)."""
    return isinstance(value, str) and value.strip().lower() == "auto"


def _find_context_key(
    config: object, key: str, depth: int = _CONTEXT_SEARCH_DEPTH
) -> Optional[int]:
    """Find a positive value at ``key``: top level first, then named sections,
    then any other nested dict (so a VLM's text_config is still reached)."""
    if not isinstance(config, dict):
        return None

    in_scope = _coerce_positive_int(config.get(key))
    if in_scope is not None:
        return in_scope
    if depth <= 0:
        return None

    for section_name in CONTEXT_SECTION_PRIORITY:
        if isinstance(config.get(section_name), dict):
            hit = _find_context_key(config[section_name], key, depth - 1)
            if hit is not None:
                return hit

    handled = set(CONTEXT_SECTION_PRIORITY)
    for name, value in config.items():
        if name not in handled and isinstance(value, dict):
            hit = _find_context_key(value, key, depth - 1)
            if hit is not None:
                return hit
    return None


def read_context_window_from_config(model_path: str) -> Optional[int]:
    """The model's ``max_position_embeddings`` from ``config.json``, or None.

    Never raises (missing / unreadable / malformed config just yields None).
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

        found = _find_context_key(config, CONTEXT_FIELD)
        if found is not None:
            return found
        logger.debug("context discovery: no %s in %s", CONTEXT_FIELD, config_path)
        return None
    except Exception as exc:  # defensive: discovery must never break loading
        logger.debug("context discovery failed for %s (%s)", model_path, exc)
        return None


def resolve_context_window(
    model_path: str, explicit: object = None
) -> Optional[int]:
    """The context window to advertise, per the opt-in rules (see module docstring)."""
    if explicit is None:
        return None
    if is_auto(explicit):
        return read_context_window_from_config(model_path)
    return _coerce_positive_int(explicit)


def format_context_window_warning(
    model_name: str, configured: int, model_max: int
) -> str:
    """A loud WARN block: ``configured`` (pinned) exceeds the real ``model_max``."""
    over_by = configured - model_max
    return "\n".join([
        "",
        "!!! WARNING: configured context_window exceeds the model's real context !!!",
        "---------------------------------------------------------------------------------",
        f"    Model '{model_name}' is configured with context_window={configured} tokens,",
        f"    but its config.json declares max_position_embeddings={model_max} tokens.",
        f"    {over_by} tokens (about {configured / model_max:.1f}x) LARGER than the model supports.",
        "    OpenArc still advertises the larger context_window in /v1/models, so",
        "    clients may send prompts the model refuses or truncates.",
        "    Lower context_window to <= its real limit, or set it to 'auto'.",
        "---------------------------------------------------------------------------------",
    ])


def check_context_window_exceeded(
    model_name: str, model_path: str, explicit: object
) -> Optional[str]:
    """Warn (return a message) when a *pinned* context_window is larger than the
    model's real max_position_embeddings; else None.

    Never warns for unset (None) or "auto"; and when config.json has no
    max_position_embeddings there is nothing real to compare, so it stays None.
    """
    if explicit is None or is_auto(explicit):
        return None
    configured = _coerce_positive_int(explicit)
    if not configured or configured <= 0:
        return None
    model_max = read_context_window_from_config(model_path)
    if model_max is None or configured <= model_max:
        return None
    return format_context_window_warning(model_name, configured, model_max)
