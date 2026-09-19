"""Layered config merging for per-request model defaults.

Precedence, highest wins:

    request-time  >  config.yaml block  >  engine/pydantic default

The tricky part is that the request contracts use ``model_fields_set`` to ask
"did the caller explicitly provide this?", and several call sites in
``routes/openai.py`` rely on that to decide whether a top-level request field
(e.g. ``request.language``) may override a value in the contract. If we built
the contract by dumping every field and re-passing it, everything would look
explicitly set and those guards would silently stop working.

``build_config`` therefore only ever passes keys that were actually provided by
either layer, leaving pydantic to supply defaults for the rest.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel

from src.server.schemas.modeling.config_blocks import (
    QWEN3_TTS_PREFIX,
    SHARED_TTS_BLOCK,
    block_applies_to,
)

T = TypeVar("T", bound=BaseModel)


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``overlay`` onto ``base``.

    Nested mappings are merged recursively. Scalars and lists are replaced
    wholesale. ``None`` values in ``overlay`` are skipped so that an omitted
    request field never clobbers a configured default.
    """
    result: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if value is None:
            continue
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def resolve_fields(
    request: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve raw mappings into the field dict to construct a contract with.

    ``None`` values in ``request`` are dropped rather than treated as explicit
    nulls, which is what makes "client omitted the field" fall through to the
    ``defaults`` layer.
    """
    clean_request = {k: v for k, v in (request or {}).items() if v is not None}
    clean_defaults = {k: v for k, v in (defaults or {}).items() if v is not None}
    return deep_merge(clean_defaults, clean_request)


def build_config(
    contract: Type[T],
    request: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> T:
    """Construct ``contract`` from a request layer over a defaults layer.

    Only keys supplied by one of the layers are passed to the constructor, so
    ``model_fields_set`` on the result contains exactly the fields that were
    genuinely provided. Pydantic fills in the contract's own defaults for
    everything else.

    Args:
        contract: The pydantic request-contract class to build.
        request: Values supplied by the caller (highest precedence). ``None``
            values are ignored.
        defaults: Values from the model's config.yaml block. ``None`` values are
            ignored.
        **extra: Keys to force, applied last (e.g. transport/mode fields the
            route derives rather than the client supplying).

    Returns:
        An instance of ``contract``.
    """
    fields = resolve_fields(request=request, defaults=defaults)
    fields.update(extra)
    # Restrict to real contract fields so a stray config key can't blow up the
    # constructor; unknown keys in a block are caught earlier by ModelLoadConfig.
    allowed = {k: v for k, v in fields.items() if k in contract.model_fields}
    return contract(**allowed)


def defaults_for_record(record: Any) -> Dict[str, Any]:
    """Resolve the config.yaml defaults that apply to a loaded ModelRecord.

    Merges every block on the record that applies to its model_type. For
    llm/vlm that is sampler_config; for a qwen3_tts_* mode it is the shared
    qwen3_tts_config plus the mode's own block (mode-specific values win).
    """
    blocks = getattr(record, "model_config_blocks", None) or {}
    if not blocks:
        return {}

    model_type = _model_type_value(record)

    applicable: Dict[str, Any] = {}
    for block_name, payload in blocks.items():
        if not isinstance(payload, Mapping):
            continue
        if not block_applies_to(block_name, model_type):
            continue
        # Applied in dict order, so a later block overrides an earlier one.
        # Merge the shared TTS block first so the mode block wins.
        if block_name == SHARED_TTS_BLOCK:
            continue
        applicable = deep_merge(applicable, payload)

    if model_type.startswith(QWEN3_TTS_PREFIX):
        shared = blocks.get(SHARED_TTS_BLOCK)
        if isinstance(shared, Mapping):
            applicable = deep_merge(shared, applicable)

    return applicable


def _model_type_value(record: Any) -> str:
    """Return a ModelRecord's model_type as a plain string."""
    model_type = getattr(record, "model_type", None)
    return model_type.value if hasattr(model_type, "value") else str(model_type or "")
