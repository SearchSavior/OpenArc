"""Unit tests for context-window discovery (pure stdlib, no heavy deps).

These exercise the field-priority resolution
(max_position_embeddings / n_positions / seq_len / seq_length / n_ctx /
sliding_window) and the explicit-override precedence used to propagate a model's
context window through /v1/models.
"""

import json

import pytest  # type: ignore[import]

from src.server.utils.context import (
    CONTEXT_FIELD_PRIORITY,
    read_context_window_from_config,
    resolve_context_window,
)


def _write_config(directory, payload) -> str:
    """Write ``payload`` as ``config.json`` inside ``directory`` and return its path."""
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(directory)


def _all_keys_config() -> dict:
    return {
        "max_position_embeddings": 128000,
        "n_positions": 40960,
        "seq_len": 8192,
        "seq_length": 16384,
        "n_ctx": 32768,
        "sliding_window": 512,
    }


def test_priority_list_matches_spec(tmp_path) -> None:
    assert CONTEXT_FIELD_PRIORITY == [
        "max_position_embeddings",
        "n_positions",
        "seq_len",
        "seq_length",
        "n_ctx",
        "sliding_window",
    ]


def test_first_present_key_wins(tmp_path) -> None:
    # max_position_embeddings is present, so it wins regardless of the (smaller)
    # values in the later-priority keys.
    path = _write_config(tmp_path, _all_keys_config())
    assert read_context_window_from_config(path) == 128000


def test_n_ctx_used_when_earlier_keys_absent(tmp_path) -> None:
    path = _write_config(tmp_path, {"n_ctx": 4096, "model_type": "llm"})
    assert read_context_window_from_config(path) == 4096


def test_sliding_window_is_last_resort(tmp_path) -> None:
    # Only a later-priority key present -> it is used.
    path = _write_config(tmp_path, {"architectures": ["X"], "sliding_window": 512})
    assert read_context_window_from_config(path) == 512


def test_earlier_key_precedes_sliding_window(tmp_path) -> None:
    # n_ctx appears before sliding_window in the priority list.
    path = _write_config(tmp_path, {"sliding_window": 512, "n_ctx": 4096})
    assert read_context_window_from_config(path) == 4096


def test_present_but_null_falls_through_to_next(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": None, "n_ctx": 4096})
    assert read_context_window_from_config(path) == 4096


def test_present_but_zero_is_skipped(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 0, "n_ctx": 4096})
    assert read_context_window_from_config(path) == 4096


def test_all_keys_zero_returns_none(tmp_path) -> None:
    path = _write_config(tmp_path, {k: 0 for k in CONTEXT_FIELD_PRIORITY})
    assert read_context_window_from_config(path) is None


def test_bool_value_is_not_a_window_size(tmp_path) -> None:
    # A lone boolean must not be mistaken for an integer window size.
    path = _write_config(tmp_path, {"max_position_embeddings": True})
    assert read_context_window_from_config(path) is None


def test_bool_key_falls_through_to_next(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": True, "n_ctx": 8192})
    assert read_context_window_from_config(path) == 8192


def test_float_value_is_coerced_to_int(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 128000.0})
    result = read_context_window_from_config(path)
    assert result == 128000
    assert isinstance(result, int)


def test_missing_config_returns_none(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert read_context_window_from_config(str(tmp_path)) is None


def test_malformed_json_returns_none(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert read_context_window_from_config(str(tmp_path)) is None


def test_non_dict_json_returns_none(tmp_path) -> None:
    (tmp_path / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert read_context_window_from_config(str(tmp_path)) is None


def test_file_model_path_uses_parent_dir(tmp_path) -> None:
    # When model_path points at a file (not a dir), config.json is looked up in
    # the containing directory, matching how a tokenizer/model file ships.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_config(model_dir, {"n_ctx": 9999})
    (model_dir / "transformer_model.json").write_text("{}", encoding="utf-8")
    assert read_context_window_from_config(str(model_dir / "transformer_model.json")) == 9999


def test_explicit_override_wins_over_derivation(tmp_path) -> None:
    path = _write_config(tmp_path, {"n_ctx": 40960})
    assert resolve_context_window(path, explicit=99999) == 99999


def test_explicit_zero_falls_through_to_derivation(tmp_path) -> None:
    path = _write_config(tmp_path, {"n_ctx": 40960})
    assert resolve_context_window(path, explicit=0) == 40960


def test_explicit_negative_falls_through_to_derivation(tmp_path) -> None:
    path = _write_config(tmp_path, {"n_ctx": 40960})
    assert resolve_context_window(path, explicit=-5) == 40960


def test_explicit_none_falls_through_to_derivation(tmp_path) -> None:
    path = _write_config(tmp_path, {"n_ctx": 40960})
    assert resolve_context_window(path, explicit=None) == 40960


# --- nested per-modality sections (multimodal / VLM configs) -----------------
#
# A flat, top-level-only scan is the bug this fix closes: Qwen2-VL / Qwen2.5-VL
# / Qwen3-VL / Qwen3.5 / Gemma3 / Mistral-Small nest max_position_embeddings
# inside text_config / language_config instead of at the top level.


def test_nested_text_config_discovered(tmp_path) -> None:
    # The realistic Qwen2.5-VL shape: the window lives under `text_config`,
    # while the (decoy) top level / vision_config carry nothing usable.
    path = _write_config(
        tmp_path,
        {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "text_config": {
                "model_type": "qwen2_5_vl_text",
                "architectures": ["Qwen2_5_VLForCausalLM"],
                "max_position_embeddings": 32768,
                "hidden_size": 4096,
            },
            "vision_config": {
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VisionTransformer"],
            },
        },
    )
    assert read_context_window_from_config(path) == 32768


def test_nested_section_beats_top_level_lower_priority(tmp_path) -> None:
    # The exact "wrong section" trap: a lower-priority key at the TOP level
    # (sliding_window) would win for a flat scan, but the higher-priority key
    # nested in text_config (max_position_embeddings) must win instead.
    path = _write_config(
        tmp_path,
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "sliding_window": 512,  # top-level, lower priority
            "text_config": {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForCausalLM"],
                "max_position_embeddings": 32768,
            },
        },
    )
    assert read_context_window_from_config(path) == 32768


def test_language_config_section_fallback(tmp_path) -> None:
    # Some families (Gemma3 / Mistral-Small 3.2) nest under `language_config`.
    path = _write_config(
        tmp_path,
        {
            "model_type": "gemma3",
            "language_config": {"max_position_embeddings": 131072},
            "vision_config": {"sliding_window": 512},
        },
    )
    assert read_context_window_from_config(path) == 131072


def test_named_section_preferred_over_arbitrary_nested(tmp_path) -> None:
    # With multiple nested dicts carrying the key, the named language section
    # wins over an arbitrary nested section -- deterministic regardless of the
    # order the keys happen to appear in the file.
    path = _write_config(
        tmp_path,
        {
            "something": {"max_position_embeddings": 111},  # arbitrary, appears first
            "text_config": {"max_position_embeddings": 888},  # named, must win
        },
    )
    assert read_context_window_from_config(path) == 888


def test_priority_within_nested_section_still_applies(tmp_path) -> None:
    # Inside a section the field priority still holds: max_position_embeddings is
    # absent, so the section's n_ctx is used (not a spurious value elsewhere).
    path = _write_config(
        tmp_path,
        {
            "model_type": "gemma3",
            "language_config": {"n_ctx": 8192},
            "vision_config": {"sliding_window": 512},
        },
    )
    assert read_context_window_from_config(path) == 8192


def test_deeply_nested_section_reached_by_fallback(tmp_path) -> None:
    # A section nested one level deeper than the named list is still reached by the
    # generic recursive fallback (so the window is never lost to extra nesting).
    path = _write_config(
        tmp_path,
        {
            "model": {"text_config": {"max_position_embeddings": 5555}},
        },
    )
    assert read_context_window_from_config(path) == 5555


def test_explicit_override_wins_over_nested_discovery(tmp_path) -> None:
    # An explicit value still short-circuits discovery, even when the discovered
    # value lives in a nested section.
    path = _write_config(
        tmp_path,
        {"text_config": {"max_position_embeddings": 131072}},
    )
    assert resolve_context_window(path, explicit=5000) == 5000


def test_nested_section_no_keys_returns_none(tmp_path) -> None:
    # A multimodal config where neither the top level nor any nested section
    # carries any of the candidate keys -> None (nothing advertised / enforced).
    path = _write_config(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "text_config": {"hidden_size": 4096, "num_attention_heads": 32},
            "vision_config": {"initializer_range": 0.02},
        },
    )
    assert read_context_window_from_config(path) is None
