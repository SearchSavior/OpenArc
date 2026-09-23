"""CLI flags generated from the pydantic request contracts.

``openarc add`` must not keep its own copy of the model-default fields: the
contracts under ``src/server/schemas/modeling/`` are the single definition of
what a request - and therefore a model default - may contain. This module
introspects those contracts and builds one ``click`` option per field, so
adding a field to a contract makes it available on ``openarc add`` with no
separate edit.

The same field name can appear in more than one contract (``temperature`` backs
both ``sampler_config`` for llm/vlm and ``qwen3_tts_config`` for Qwen3-TTS).
Because ``--model-type`` is not known until after parsing, one flag is declared
per field name and the destination is resolved after parse: the flag lands in
whichever block the chosen model_type uses. A flag with no home for that
model_type is rejected rather than silently dropped.

Mapping to the config.yaml grouping:

- every field of a contract in ``BLOCK_CONTRACTS`` becomes a flag written to
  that block (``sampler_config``, ``kokoro_config``, ...), except
  ``REQUEST_ONLY_FIELDS`` (per-request data), the config_blocks
  ``LOAD_CONFIG_FIELDS`` (load-time plumbing), and the speculative-decoding
  pair ``openarc add`` already declares as dedicated load flags
- every field of ``SchedulerConfigSchema`` becomes a flag written to the
  ``scheduler_config`` sibling key
- ``runtime_config`` stays a hand-declared JSON string on the command: it is a
  free-form dict of OpenVINO properties with no contract to generate flags from

For --help, ``option_groups()`` produces rich_click's own OPTION_GROUPS layout,
so each config.yaml key becomes one help panel.
"""
from __future__ import annotations

import enum
import json
import re
import types
from typing import Any, Dict, List, Tuple, Union, get_args, get_origin

import click
from pydantic import BaseModel, ValidationError

from src.server.schemas.modeling.config_blocks import (
    BLOCK_CONTRACTS,
    LOAD_CONFIG_FIELDS,
    QWEN3_TTS_PREFIX,
    REQUEST_ONLY_FIELDS,
    SAMPLER_BLOCK,
    SAMPLER_MODEL_TYPES,
    SHARED_TTS_BLOCK,
    model_block_name,
    validate_block,
)
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import (
    SchedulerConfigSchema,
)

SCHEDULER_DEST = "scheduler_config"
LOAD_CONFIG_DEST = "load_config"

# Contract fields `openarc add` already exposes as dedicated load_config flags.
# They exist in OVGenAI_GenConfig as request-time overrides, but at add time
# speculative decoding is configured on load_config (ModelLoadConfig), so they
# are not re-declared as block flags and cannot land in sampler_config.
ADD_DECLARED_FIELDS = frozenset(
    {"num_assistant_tokens", "assistant_confidence_threshold"}
)

# Marker for fields rendered as a JSON-string flag (dict-typed contract fields
# such as chat_template_kwargs, which have no scalar CLI spelling).
_JSON = "json"

# Destination -> contract.
_DEST_CONTRACTS: Dict[str, type[BaseModel]] = {
    name: contract for name, (contract, _) in BLOCK_CONTRACTS.items()
}
_DEST_CONTRACTS[SCHEDULER_DEST] = SchedulerConfigSchema

# --help/registry order: the request-default blocks grouped by model family,
# then scheduler.
_GROUP_ORDER: List[str] = [
    SAMPLER_BLOCK,
    SHARED_TTS_BLOCK,
    "qwen3_tts_custom_voice_config",
    "qwen3_tts_voice_design_config",
    "qwen3_tts_voice_clone_config",
    "kokoro_config",
    "qwen3_asr_config",
    SCHEDULER_DEST,
]

_WHITESPACE = re.compile(r"\s+")


class ConfigOptionError(ValueError):
    """A provided flag is invalid, or does not apply to the chosen model_type."""


def kebab(field: str) -> str:
    return field.replace("_", "-")


def _scalar(annotation: Any) -> Any:
    """Reduce an annotation to what a single CLI flag can carry.

    ``Optional[X]`` and ``X | None`` unwrap to ``X``. Returns the ``_JSON``
    marker for dict fields, and None for anything else (lists, nested models),
    which cannot be expressed as one flag.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _scalar(args[0]) if len(args) == 1 else None
    if annotation is dict or origin is dict:
        return _JSON
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation
    if annotation in (bool, int, float, str):
        return annotation
    return None


def _help_text(info: Any) -> str:
    return _WHITESPACE.sub(" ", info.description or "").strip()


def exposed_fields(dest: str) -> List[str]:
    """Fields of a destination that become CLI flags, in contract order."""
    return [
        name
        for name in _DEST_CONTRACTS[dest].model_fields
        if name not in REQUEST_ONLY_FIELDS
        and name not in LOAD_CONFIG_FIELDS
        and name not in ADD_DECLARED_FIELDS
    ]


def field_registry() -> Dict[str, Tuple[Any, str]]:
    """Map every CLI-exposed field to (scalar type | _JSON, help text).

    Fields are deduped by name in group order, so a name shared by two
    contracts takes the first contract's type and description.
    """
    registry: Dict[str, Tuple[Any, str]] = {}
    for dest in _GROUP_ORDER:
        contract = _DEST_CONTRACTS[dest]
        for name in exposed_fields(dest):
            if name in registry:
                continue
            info = contract.model_fields[name]
            scalar = _scalar(info.annotation)
            if scalar is None:
                raise ConfigOptionError(
                    f"Cannot expose contract field '{dest}.{name}' "
                    f"({info.annotation!r}) as a CLI flag. Teach "
                    f"config_options.py to render it, or exclude it via "
                    f"REQUEST_ONLY_FIELDS."
                )
            registry[name] = (scalar, _help_text(info))
    return registry


def build_config_options() -> List[click.Option]:
    """Build one click option per field defined by the contracts."""
    options: List[click.Option] = []
    for name, (scalar, help_text) in field_registry().items():
        kebab_name = kebab(name)
        if scalar is bool:
            # Plain flag: absent means "not provided" (None) so nothing is
            # written and the contract default applies. Passing it sets True;
            # there is no --no- spelling.
            params = [f"--{kebab_name}"]
            attrs: Dict[str, Any] = {
                "is_flag": True,
                "default": None,
                "help": help_text,
            }
        elif scalar == _JSON:
            params = [f"--{kebab_name}"]
            attrs = {
                "default": None,
                "help": f"{help_text} (JSON object)".strip(),
            }
        else:
            params = [f"--{kebab_name}"]
            attrs = {"default": None, "help": help_text}
            if isinstance(scalar, type) and issubclass(scalar, enum.Enum):
                attrs["type"] = click.Choice([member.value for member in scalar])
            else:
                attrs["type"] = scalar
        options.append(click.Option(params, **attrs))
    return options


def config_options(command):
    """Decorator attaching every contract-derived flag to a command.

    Params are appended in reverse: click reverses ``__click_params__`` when it
    builds the Command, so this is what lands them in registry order.
    """
    params = getattr(command, "__click_params__", None)
    if params is None:
        params = []
        command.__click_params__ = params
    params.extend(reversed(build_config_options()))
    return command


def _group_label(dest: str) -> str:
    """Human label for a help group, naming the model types the key serves."""
    if dest == SAMPLER_BLOCK:
        return f"{dest}   ·   {', '.join(sorted(SAMPLER_MODEL_TYPES))}"
    if dest == SHARED_TTS_BLOCK:
        return f"{dest}   ·   {QWEN3_TTS_PREFIX}*"
    if dest == SCHEDULER_DEST:
        return f"{dest}   ·   all model types"
    required = BLOCK_CONTRACTS[dest][1]
    return f"{dest}   ·   {required}" if required else dest


def option_groups(load_options: List[str]) -> List[Dict[str, Any]]:
    """Build rich_click's OPTION_GROUPS value for `openarc add`.

    One group per config.yaml key, each listing every setting the key accepts,
    so a field shared by two keys appears in both. ``deduplicate: False`` is
    what lets rich_click render the same option in more than one panel.

    Args:
        load_options: Parameter names of the hand-declared load_config flags,
            plus click's own ``help``.
    """
    groups: List[Dict[str, Any]] = [
        {"name": LOAD_CONFIG_DEST, "options": list(load_options)}
    ]
    shared_tts = set(exposed_fields(SHARED_TTS_BLOCK))
    for dest in _GROUP_ORDER:
        fields = exposed_fields(dest)
        if dest != SHARED_TTS_BLOCK and dest.startswith(QWEN3_TTS_PREFIX):
            # A Qwen3-TTS mode contract subclasses the shared TTS contract, so
            # it carries every shared field. The shared panel already lists
            # them; this panel shows only the mode's extras.
            fields = [name for name in fields if name not in shared_tts]
        groups.append(
            {
                "name": _group_label(dest),
                "options": fields,
                "deduplicate": False,
            }
        )
    return groups


def allowed_destinations(model_type: str) -> List[str]:
    """Destination blocks a model_type may be configured with, in claim order.

    A Qwen3-TTS mode contract subclasses the shared TTS contract, so the shared
    block claims the fields they have in common and the mode block keeps only
    its extras.
    """
    destinations: List[str] = []
    if model_type in SAMPLER_MODEL_TYPES:
        destinations.append(SAMPLER_BLOCK)
    if model_type.startswith(QWEN3_TTS_PREFIX):
        destinations.append(SHARED_TTS_BLOCK)
    mode_block = model_block_name(model_type)
    if mode_block:
        destinations.append(mode_block)
    destinations.append(SCHEDULER_DEST)
    return destinations


def _destinations_by_field(model_type: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for dest in allowed_destinations(model_type):
        for name in exposed_fields(dest):
            mapping.setdefault(name, dest)
    return mapping


def _plain(value: Any) -> Any:
    """Reduce enum members to their serializable values."""
    return value.value if isinstance(value, enum.Enum) else value


def _validated_scheduler(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    try:
        validated = SchedulerConfigSchema(**payload)
    except ValidationError as exc:
        raise ConfigOptionError(f"scheduler_config has invalid values: {exc}") from exc
    return {key: _plain(getattr(validated, key)) for key in payload}


def _parse_json_flags(
    provided: Dict[str, Any], registry: Dict[str, Tuple[Any, str]]
) -> Dict[str, Any]:
    """Parse values of JSON-rendered flags; pass everything else through."""
    parsed: Dict[str, Any] = {}
    for name, value in provided.items():
        if registry.get(name, (None, ""))[0] != _JSON:
            parsed[name] = value
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigOptionError(
                f"--{kebab(name)} expects a JSON object: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ConfigOptionError(
                f"--{kebab(name)} expects a JSON object, got {type(decoded).__name__}"
            )
        parsed[name] = decoded
    return parsed


def resolve_config_values(
    model_type: str, values: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Route provided flag values to the config blocks they belong to.

    Args:
        model_type: The ``--model-type`` chosen by the operator.
        values: Raw values keyed by field name; None means "not provided".

    Returns:
        ``(scheduler_config, blocks)`` where scheduler_config is written as a
        sibling of ``load_config`` and blocks map block name -> validated
        authored keys.

    Raises:
        ConfigOptionError: If a flag does not apply to model_type, or a value
            fails contract validation.
    """
    mapping = _destinations_by_field(model_type)
    provided = {name: value for name, value in values.items() if value is not None}
    provided = _parse_json_flags(provided, field_registry())

    unknown = sorted(set(provided) - set(mapping))
    if unknown:
        flags = ", ".join(f"--{kebab(name)}" for name in unknown)
        valid = ", ".join(f"--{kebab(name)}" for name in sorted(mapping))
        raise ConfigOptionError(
            f"Option(s) {flags} do not apply to model_type '{model_type}'. "
            f"Valid config options for this model type: {valid or 'none'}."
        )

    payloads: Dict[str, Dict[str, Any]] = {}
    for name, value in provided.items():
        payloads.setdefault(mapping[name], {})[name] = value

    scheduler = _validated_scheduler(payloads.pop(SCHEDULER_DEST, {}))

    blocks: Dict[str, Dict[str, Any]] = {}
    for dest, payload in payloads.items():
        try:
            resolved = validate_block(dest, payload, model_type)
        except ValueError as exc:
            raise ConfigOptionError(str(exc)) from exc
        if resolved:
            blocks[dest] = {key: _plain(value) for key, value in resolved.items()}

    return scheduler, blocks
