import pytest  # type: ignore[import]

from src.engine.openvino.qwen3_asr.qwen3_asr_utils import (
    LANGUAGE_CODE_TO_NAME,
    SUPPORTED_LANGUAGES,
    resolve_language_name,
    validate_language,
)


def test_resolve_language_name_maps_iso_639_1_code() -> None:
    assert resolve_language_name("en") == "English"
    assert resolve_language_name("zh") == "Chinese"


def test_resolve_language_name_code_is_case_insensitive_and_trimmed() -> None:
    assert resolve_language_name("ZH") == "Chinese"
    assert resolve_language_name(" ja ") == "Japanese"


def test_resolve_language_name_passes_through_full_names() -> None:
    assert resolve_language_name("english") == "English"
    assert resolve_language_name("cHINese") == "Chinese"


def test_resolve_language_name_handles_non_iso_639_1_special_cases() -> None:
    # Cantonese has no ISO-639-1 code; Filipino accepts both 639-1 (tl) and 639-3 (fil).
    assert resolve_language_name("yue") == "Cantonese"
    assert resolve_language_name("tl") == "Filipino"
    assert resolve_language_name("fil") == "Filipino"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_resolve_language_name_rejects_empty(value) -> None:
    with pytest.raises(ValueError):
        resolve_language_name(value)


def test_every_mapped_code_resolves_to_a_supported_language() -> None:
    # Guards against drift if SUPPORTED_LANGUAGES is edited without updating the map.
    for code, name in LANGUAGE_CODE_TO_NAME.items():
        assert name in SUPPORTED_LANGUAGES, f"{code} -> {name} not in SUPPORTED_LANGUAGES"
        # Resolved canonical names must pass validation.
        validate_language(resolve_language_name(code))
