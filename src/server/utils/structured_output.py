"""Translate an OpenAI ``response_format`` into OpenVINO GenAI structured output.

The OV GenAI engines clamp decoding with ``StructuredOutputConfig`` (xgrammar
backend); this module is the pure half of that wiring, mapping the request field
to the config's kwargs. Building the OpenVINO object is the engine's job.
"""
import json
from typing import Any, Dict, Mapping

UNCONSTRAINED_TYPES = frozenset({"text", "text/plain"})

NOT_AN_OBJECT = "response_format must be an object with a 'type' key, got {kind}"
JSON_SCHEMA_FIELD_REQUIRED = (
    "response_format type 'json_schema' requires a 'json_schema' field"
)
JSON_SCHEMA_INVALID = "response_format.json_schema must be an object or a JSON string"
UNSUPPORTED_TYPE = (
    "unsupported response_format type {fmt_type!r}; "
    "supported types: json_schema, json_object, text"
)


class ResponseFormatError(ValueError):
    """A response_format that cannot be mapped to a grammar constraint."""


def structured_output_kwargs(
    response_format: Mapping[str, Any] | None,
) -> Dict[str, str] | None:
    """Return ``StructuredOutputConfig`` kwargs for an OpenAI ``response_format``.

    ``json_schema`` (OpenAI's ``{name, strict, schema}`` wrapper or a bare
    schema) and ``json_object`` produce a grammar constraint; ``text``/None
    produce none. Anything else raises ``ResponseFormatError`` so a caller that
    asked for clamped JSON never silently receives free-form text.
    """
    if not response_format:
        return None
    if not isinstance(response_format, Mapping):
        raise ResponseFormatError(NOT_AN_OBJECT.format(kind=type(response_format).__name__))

    fmt_type = response_format.get("type")
    if fmt_type is None or fmt_type in UNCONSTRAINED_TYPES:
        return None
    if fmt_type == "json_object":
        return {"json_schema": json.dumps({"type": "object"})}
    if fmt_type == "json_schema":
        return {"json_schema": _json_schema_string(response_format)}

    raise ResponseFormatError(UNSUPPORTED_TYPE.format(fmt_type=fmt_type))


def _json_schema_string(response_format: Mapping[str, Any]) -> str:
    """Extract the JSON schema string from a ``json_schema`` response_format."""
    json_schema = response_format.get("json_schema")
    if json_schema is None:
        raise ResponseFormatError(JSON_SCHEMA_FIELD_REQUIRED)
    if isinstance(json_schema, Mapping):
        # OpenAI nests the schema under "schema"; llama.cpp and others pass the
        # schema itself. "schema" is not a JSON Schema keyword, so a top-level
        # "schema" key is unambiguously the wrapper.
        json_schema = json_schema.get("schema", json_schema)
    if isinstance(json_schema, str):
        return json_schema
    if isinstance(json_schema, Mapping):
        return json.dumps(json_schema)
    raise ResponseFormatError(JSON_SCHEMA_INVALID)
