"""YAML config loading, env interpolation, and strict validation."""
from pathlib import Path

import pytest  # type: ignore[import]
import yaml
from pydantic import ValidationError

from src.cli.modules.server_config import ServerConfig
from src.server.schemas.registration import ModelLoadConfig


def _cfg(tmp_path: Path) -> ServerConfig:
    return ServerConfig(config_file=tmp_path / "config.yaml")


def _write(cfg: ServerConfig, payload: dict) -> None:
    cfg.config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ---- load / save ----


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert _cfg(tmp_path).load_config() == {}


def test_load_parses_yaml(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, {"server": {"port": 8000}, "models": {}})
    assert cfg.load_config()["server"]["port"] == 8000


def test_load_returns_empty_on_invalid_yaml(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.config_file.write_text("models: [unclosed\n", encoding="utf-8")
    assert cfg.load_config() == {}


def test_load_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert cfg.load_config() == {}


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    payload = {"models": {"a": {"load_config": {"model_type": "llm", "device": "CPU"}}}}
    cfg.save_config(payload)
    assert cfg.load_config() == payload


def test_save_is_noop_when_unchanged(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    payload = {"models": {"a": {"load_config": {"device": "CPU"}}}}
    cfg.save_config(payload)
    first_mtime = cfg.config_file.stat().st_mtime_ns
    cfg.save_config(payload)
    assert cfg.config_file.stat().st_mtime_ns == first_mtime


# ---- env interpolation ----


def test_interpolation_expands_set_variable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MY_MODELS", "a,b")
    cfg = _cfg(tmp_path)
    _write(cfg, {"startup_models": "${MY_MODELS}"})
    assert cfg.load_config()["startup_models"] == "a,b"


def test_interpolation_uses_default_when_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    cfg = _cfg(tmp_path)
    _write(cfg, {"startup_models": "${MISSING_VAR:-fallback}"})
    assert cfg.load_config()["startup_models"] == "fallback"


def test_interpolation_unset_without_default_becomes_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    cfg = _cfg(tmp_path)
    _write(cfg, {"startup_models": "${MISSING_VAR}"})
    assert cfg.load_config()["startup_models"] == ""


def test_interpolation_recurses_into_nested_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEV", "GPU.0")
    cfg = _cfg(tmp_path)
    _write(cfg, {"models": {"a": {"load_config": {"device": "${DEV}"}}}})
    assert cfg.load_config()["models"]["a"]["load_config"]["device"] == "GPU.0"


def test_interpolation_does_not_expand_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY", "expanded")
    cfg = _cfg(tmp_path)
    _write(cfg, {"${KEY}": "value"})
    assert cfg.load_config() == {"${KEY}": "value"}


# ---- nested entry access ----


def test_get_model_config_flattens_nested_entry(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(
        cfg,
        {
            "models": {
                "a": {
                    "load_config": {"model_type": "llm", "device": "CPU"},
                    "sampler_config": {"temperature": 0.7},
                }
            }
        },
    )
    flat = cfg.get_model_config("a")
    assert flat["model_type"] == "llm"
    assert flat["device"] == "CPU"


def test_get_model_config_still_accepts_flat_entry(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, {"models": {"a": {"model_type": "llm", "device": "CPU"}}})
    assert cfg.get_model_config("a")["model_type"] == "llm"


def test_get_model_load_config_injects_name_from_key(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, {"models": {"real-name": {"load_config": {"model_type": "llm", "engine": "ovgenai", "device": "CPU", "model_path": "/p"}}}})
    assert cfg.get_model_load_config("real-name").model_name == "real-name"


def test_get_model_load_config_returns_none_for_unknown(tmp_path: Path) -> None:
    _write(_cfg(tmp_path), {"models": {}})
    assert _cfg(tmp_path).get_model_load_config("nope") is None


# ---- strict validation ----


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(
        cfg,
        {
            "models": {
                "a": {
                    "load_config": {
                        "model_type": "llm",
                        "engine": "ovgenai",
                        "device": "CPU",
                        "model_path": "/p",
                        "bogus_key": 1,
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="Invalid configuration"):
        cfg.get_model_load_config("a")


def test_block_model_type_mismatch_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(
        cfg,
        {
            "models": {
                "a": {
                    "load_config": {
                        "model_type": "qwen3_tts_voice_clone",
                        "engine": "openvino",
                        "device": "CPU",
                        "model_path": "/p",
                    },
                    "qwen3_tts_custom_voice_config": {"speaker": "x"},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="qwen3_tts_custom_voice_config requires"):
        cfg.get_model_load_config("a")


def test_sampler_config_rejected_for_non_llm(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(
        cfg,
        {
            "models": {
                "a": {
                    "load_config": {
                        "model_type": "kokoro",
                        "engine": "openvino",
                        "device": "CPU",
                        "model_path": "/p",
                    },
                    "sampler_config": {"temperature": 0.7},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="sampler_config is only valid"):
        cfg.get_model_load_config("a")


def test_save_model_config_writes_nested_shape(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.save_model_config("a", {"model_type": "llm", "device": "CPU"})
    assert yaml.safe_load(cfg.config_file.read_text())["models"]["a"] == {
        "load_config": {"model_type": "llm", "device": "CPU"}
    }


def test_save_model_entry_preserves_sibling_blocks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.save_model_entry(
        "a",
        {"load_config": {"model_type": "llm"}, "sampler_config": {"temperature": 0.7}},
    )
    entry = yaml.safe_load(cfg.config_file.read_text())["models"]["a"]
    assert entry["sampler_config"] == {"temperature": 0.7}
