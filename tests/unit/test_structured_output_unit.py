import json
from typing import Any, Mapping

import pytest  # type: ignore[import]

from src.server.utils.structured_output import (
    JSON_SCHEMA_FIELD_REQUIRED,
    JSON_SCHEMA_INVALID,
    NOT_AN_OBJECT,
    ResponseFormatError,
    UNSUPPORTED_TYPE,
    structured_output_kwargs,
)

OBJECT_SCHEMA = {"type": "object", "properties": {"contacts": {"type": "array"}}}


@pytest.mark.parametrize(
    "response_format",
    [None, {}, {"type": "text"}, {"type": "text/plain"}],
)
def test_unconstrained_formats_return_none(
    response_format: Mapping[str, Any] | None,
) -> None:
    assert structured_output_kwargs(response_format) is None


def test_json_object_clamps_to_any_object() -> None:
    assert structured_output_kwargs({"type": "json_object"}) == {
        "json_schema": json.dumps({"type": "object"})
    }


@pytest.mark.parametrize(
    "json_schema",
    [
        {"name": "contact", "strict": True, "schema": OBJECT_SCHEMA},
        OBJECT_SCHEMA,
        json.dumps(OBJECT_SCHEMA),
        {"schema": json.dumps(OBJECT_SCHEMA)},
    ],
    ids=["openai-wrapper", "bare-schema", "pre-encoded", "encoded-in-wrapper"],
)
def test_json_schema_is_clamped(json_schema: Any) -> None:
    result = structured_output_kwargs(
        {"type": "json_schema", "json_schema": json_schema}
    )
    assert json.loads(result["json_schema"]) == OBJECT_SCHEMA


def test_missing_json_schema_field_raises() -> None:
    with pytest.raises(ResponseFormatError) as exc:
        structured_output_kwargs({"type": "json_schema"})
    assert str(exc.value) == JSON_SCHEMA_FIELD_REQUIRED


def test_invalid_json_schema_raises() -> None:
    with pytest.raises(ResponseFormatError) as exc:
        structured_output_kwargs({"type": "json_schema", "json_schema": 42})
    assert str(exc.value) == JSON_SCHEMA_INVALID


def test_unsupported_type_raises() -> None:
    with pytest.raises(ResponseFormatError) as exc:
        structured_output_kwargs({"type": "regex"})
    assert str(exc.value) == UNSUPPORTED_TYPE.format(fmt_type="regex")


def test_non_mapping_raises() -> None:
    with pytest.raises(ResponseFormatError) as exc:
        structured_output_kwargs("json_object")
    assert str(exc.value) == NOT_AN_OBJECT.format(kind="str")
