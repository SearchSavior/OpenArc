"""Unit tests for the opt-in context-window advertisement (pure stdlib).

Exercises the three new rules:
- only ``max_position_embeddings`` is read from ``config.json`` (the other names
  some exporters use for the same concept are ignored);
- advertisement is opt-in (unset -> nothing; ``"auto"`` -> discover; int -> as-is);
- a *pinned* value above the model's real limit yields a loud warning.
"""

import json

from src.server.utils.context import (
    CONTEXT_FIELD,
    _coerce_positive_int,
    check_context_window_exceeded,
    format_context_window_warning,
    is_auto,
    read_context_window_from_config,
    resolve_context_window,
)


def _write_config(directory, payload) -> str:
    """Write ``payload`` as ``config.json`` inside ``directory``; return its path."""
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(directory)


def _dir_without_config(directory) -> str:
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return str(directory)


# --- only max_position_embeddings is read ------------------------------

def test_only_max_position_embeddings_key(tmp_path) -> None:
    assert CONTEXT_FIELD == "max_position_embeddings"


def test_ignores_other_names(tmp_path) -> None:
    # n_positions / seq_len / seq_length / n_ctx / sliding_window are no longer read.
    path = _write_config(
        tmp_path,
        {
            "n_positions": 40960,
            "seq_len": 8192,
            "seq_length": 16384,
            "n_ctx": 32768,
            "sliding_window": 512,
        },
    )
    assert read_context_window_from_config(path) is None
    assert resolve_context_window(path, "auto") is None


def test_reads_flat_max_position_embeddings(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 128000})
    assert read_context_window_from_config(path) == 128000


def test_zero_or_null_max_ignored(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 0, "text_config": {}})
    assert read_context_window_from_config(path) is None


def test_nested_text_config_discovered(tmp_path) -> None:
    # Multimodal models nest max_position_embeddings under text_config; the
    # top-level sliding_window is a decoy that is simply never read.
    path = _write_config(
        tmp_path,
        {
            "model_type": "qwen2_5_vl",
            "sliding_window": 512,
            "text_config": {"max_position_embeddings": 131072},
            "vision_config": {"model_type": "qwen2_5_vl"},
        },
    )
    assert read_context_window_from_config(path) == 131072


def test_named_section_prefers_language_over_arbitrary(tmp_path) -> None:
    # text_config wins over an arbitrary nested dict, regardless of file order.
    path = _write_config(
        tmp_path,
        {
            "something": {"max_position_embeddings": 111},
            "text_config": {"max_position_embeddings": 888},
        },
    )
    assert read_context_window_from_config(path) == 888


def test_deeply_nested_section_reached(tmp_path) -> None:
    path = _write_config(tmp_path, {"model": {"text_config": {"max_position_embeddings": 5555}}})
    assert read_context_window_from_config(path) == 5555


def test_file_model_path_uses_parent_dir(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_config(model_dir, {"max_position_embeddings": 9999})
    (model_dir / "transformer_model.json").write_text("{}", encoding="utf-8")
    got = read_context_window_from_config(str(model_dir / "transformer_model.json"))
    assert got == 9999


def test_missing_config_returns_none(tmp_path) -> None:
    assert read_context_window_from_config(_dir_without_config(tmp_path)) is None


def test_malformed_json_returns_none(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert read_context_window_from_config(str(tmp_path)) is None


def test_non_dict_json_returns_none(tmp_path) -> None:
    (tmp_path / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert read_context_window_from_config(str(tmp_path)) is None


def test_float_is_truncated(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 128000.0})
    result = read_context_window_from_config(path)
    assert result == 128000
    assert isinstance(result, int)


# --- opt-in resolution ------------------------------------------------

def test_unset_is_opt_out(tmp_path) -> None:
    # Unset -> nothing is read or advertised (config.json is not even touched).
    path = _write_config(tmp_path, {"max_position_embeddings": 131072})
    assert resolve_context_window(path, None) is None


def test_auto_discovers(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 32768})
    assert resolve_context_window(path, "auto") == 32768
    assert resolve_context_window(path, "AUTO") == 32768
    assert resolve_context_window(path, "  auto  ") == 32768


def test_auto_with_no_config_returns_none(tmp_path) -> None:
    assert resolve_context_window(_dir_without_config(tmp_path), "auto") is None


def test_explicit_int_used_as_is(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_window(path, 5000) == 5000
    assert resolve_context_window(path, 8192) == 8192


def test_non_positive_explicit_is_none(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_window(path, 0) is None
    assert resolve_context_window(path, -5) is None


def test_garbage_string_is_none(tmp_path) -> None:
    # An unparseable, non-"auto" string advertises nothing (opt-in stays off).
    path = _write_config(tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_window(path, "garbage") is None


def test_numeric_string_is_coerced(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_window(path, "32768") == 32768


# --- coercion + auto helpers ------------------------------------------

def test_coerce_positive_int() -> None:
    assert _coerce_positive_int(10) == 10
    assert _coerce_positive_int(0) is None
    assert _coerce_positive_int(-1) is None
    assert _coerce_positive_int(None) is None
    assert _coerce_positive_int(True) is None       # a lone bool is not a window size
    assert _coerce_positive_int(12.9) == 12         # floats are truncated
    assert _coerce_positive_int("32768") == 32768
    assert _coerce_positive_int("auto") is None
    assert _coerce_positive_int("  16 384 ") is None


def test_is_auto() -> None:
    assert is_auto("auto") is True
    assert is_auto("AUTO") is True
    assert is_auto(" auto ") is True
    assert is_auto("autoish") is False
    assert is_auto(42) is False
    assert is_auto(None) is False


# --- loud warning for an over-large pin -------------------------------

def test_warns_when_pinned_above_real(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 32768})
    msg = check_context_window_exceeded("model", path, 131072)
    assert msg is not None
    assert "131072" in msg and "32768" in msg
    assert msg.strip().startswith("!!!")


def test_no_warn_when_pinned_at_or_below(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 32768})
    assert check_context_window_exceeded("model", path, 32768) is None  # equal
    assert check_context_window_exceeded("model", path, 16000) is None  # below


def test_no_warn_when_unset_or_auto(tmp_path) -> None:
    path = _write_config(tmp_path, {"max_position_embeddings": 32768})
    assert check_context_window_exceeded("model", path, None) is None
    assert check_context_window_exceeded("model", path, "auto") is None


def test_no_warn_when_real_limit_unknown(tmp_path) -> None:
    # Pinned value but no max_position_embeddings to compare against -> no
    # warning (an honest "unknown", not a false positive).
    path = _write_config(tmp_path, {"n_ctx": 4096})
    assert check_context_window_exceeded("model", path, 131072) is None


def test_numeric_string_pin_still_warns(tmp_path) -> None:
    # A pin that arrives as a string (e.g. quoted in config.yaml) still warns.
    path = _write_config(tmp_path, {"max_position_embeddings": 32768})
    assert check_context_window_exceeded("model", path, "131072") is not None


def test_format_contains_model_and_ratio() -> None:
    msg = format_context_window_warning("qwen-demo", 131072, 32768)
    assert "qwen-demo" in msg
    assert "131072" in msg and "32768" in msg
    assert "4.0x" in msg
