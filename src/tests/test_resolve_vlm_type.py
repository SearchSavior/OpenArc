import json

import pytest  # type: ignore[import]

from src.server.utils.resolve_vlm_type import resolve_vlm_vision_token


def _write_config(model_dir, config) -> None:
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_resolves_token_from_architectures_list(tmp_path) -> None:
    _write_config(
        tmp_path,
        {"architectures": ["Qwen2_5_VLForConditionalGeneration"]},
    )

    assert resolve_vlm_vision_token(str(tmp_path)) == "<|vision_start|><|image_pad|><|vision_end|>"


def test_resolves_token_from_architectures_string(tmp_path) -> None:
    _write_config(tmp_path, {"architectures": "Gemma4ForConditionalGeneration"})

    assert resolve_vlm_vision_token(str(tmp_path)) == "<|image><|image|><image|>"


def test_raw_scan_fallback_ignores_model_type_key(tmp_path) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "Gemma3ForConditionalGeneration",
            "text_config": {"auto_map": "Qwen3VLForConditionalGeneration"},
        },
    )

    assert resolve_vlm_vision_token(str(tmp_path)) == "<|vision_start|><|image_pad|><|vision_end|>"


def test_vlm_type_does_not_fallback_when_config_has_no_known_architecture(tmp_path) -> None:
    _write_config(tmp_path, {"architectures": ["UnknownForConditionalGeneration"]})

    with pytest.raises(ValueError, match="Supported architectures"):
        resolve_vlm_vision_token(str(tmp_path))


def test_missing_config_raises_clear_error(tmp_path) -> None:
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Could not read VLM config.json"):
        resolve_vlm_vision_token(str(tmp_path))


def test_unknown_architecture_raises_clear_error(tmp_path) -> None:
    _write_config(tmp_path, {"architectures": ["UnknownForConditionalGeneration"]})

    with pytest.raises(ValueError, match="Supported architectures"):
        resolve_vlm_vision_token(str(tmp_path))


def test_config_model_type_is_ignored_even_when_it_contains_supported_architecture(tmp_path) -> None:
    _write_config(tmp_path, {"model_type": "Gemma4ForConditionalGeneration"})

    with pytest.raises(ValueError, match="Supported architectures"):
        resolve_vlm_vision_token(str(tmp_path))
