"""End-to-end coverage of the loud context-window warning the operator sees.

The same over-sized-pin check (src.server.utils.context.check_context_window_
exceeded) surfaces in three places; each is exercised here:

- the ``openarc add --context-window`` value must parse ``auto`` / an int
  (ContextWindowType) and be written to config.yaml;
- ``openarc serve`` pre-flights each startup model (``_startup_context_window_
  warning``) using the real ServerConfig -> ModelLoadConfig path;
- ``openarc load`` relays it back from ``POST /openarc/load`` to the CLI.
"""

import asyncio
import json
from pathlib import Path

import pytest  # type: ignore[import]
import yaml

from src.cli.groups.add import ContextWindowType
from src.cli.groups.serve import _startup_context_window_warning
from src.cli.modules.server_config import ServerConfig
import src.server.routes.openarc as openarc_module
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelType,
)


def _make_model_dir(parent: Path, name: str, max_position_embeddings: int) -> str:
    d = parent / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"max_position_embeddings": max_position_embeddings}),
        encoding="utf-8",
    )
    return str(d)


# --- the --context-window ValueParser (openarc add) ----------------------

def test_context_window_type_parses_auto() -> None:
    assert ContextWindowType().convert("auto", None, None) == "auto"
    assert ContextWindowType().convert("AUTO", None, None) == "auto"


def test_context_window_type_parses_int() -> None:
    assert ContextWindowType().convert("32768", None, None) == 32768


def test_context_window_type_rejects_non_positive_and_garbage() -> None:
    for bad in ("0", "-5", "abc"):
        with pytest.raises(Exception):
            ContextWindowType().convert(bad, None, None)


# --- openarc serve pre-flight, through the real ServerConfig -------------

def _write_config(config_dir: Path, models: dict) -> Path:
    config_path = config_dir / "config.yaml"
    payload = {
        "server": {"host": "localhost", "port": 8000},
        "models": models,
    }
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return config_path


def test_serve_preflight_warns_for_overlarge_pin(tmp_path: Path) -> None:
    # All three models share a real 32768-token limit in config.json.
    d_over = _make_model_dir(tmp_path / "models", "over", 32768)
    d_auto = _make_model_dir(tmp_path / "models", "auto", 32768)
    d_unset = _make_model_dir(tmp_path / "models", "unset", 32768)

    def base(model_path: str, extra: dict) -> dict:
        return {
            "model_name": "x",
            "model_path": model_path,
            "model_type": "llm",
            "engine": "ovgenai",
            "device": "CPU",
            **extra,
        }

    models = {
        "over": {"load_config": base(d_over, {"context_window": 131072})},
        "auto": {"load_config": base(d_auto, {"context_window": "auto"})},
        "unset": {"load_config": base(d_unset, {})},  # no context_window key
    }
    config_path = _write_config(tmp_path, models)
    server_config = ServerConfig(config_file=config_path)

    warning = _startup_context_window_warning(server_config, "over")
    assert warning is not None
    assert "131072" in warning and "32768" in warning

    # "auto" and an unset value never warn.
    assert _startup_context_window_warning(server_config, "auto") is None
    assert _startup_context_window_warning(server_config, "unset") is None


# --- openarc load relays the warning back to the CLI ---------------------

class _RelayRegistry:
    """Minimal stand-in for ModelRegistry exercising only what /openarc/load
    uses: register_load (returns an id) and get_context_window_warning (relay)."""

    def __init__(self, warning: str | None) -> None:
        self._warning = warning

    async def register_load(self, load_config: ModelLoadConfig) -> str:
        return "model-id"

    def get_context_window_warning(self, model_name: str) -> str | None:
        return self._warning


def _load_config(name: str, model_path: str, context_window: int | str | None = None):
    kwargs = dict(
        model_path=model_path,
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )
    if context_window is not None:
        kwargs["context_window"] = context_window
    return ModelLoadConfig(**kwargs)


def test_openarc_load_relays_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    warning = "!!! WARNING: configured context_window exceeds the model's real context !!!"
    reg = _RelayRegistry(warning)
    monkeypatch.setattr(openarc_module, "_registry", reg)

    async def _run():
        return await openarc_module.load_model(_load_config("over", str(tmp_path)))

    response = asyncio.run(_run())
    assert response["warning"] == warning
    assert response["status"] == "loaded"


def test_openarc_load_omits_key_without_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = _RelayRegistry(None)
    monkeypatch.setattr(openarc_module, "_registry", reg)

    async def _run():
        return await openarc_module.load_model(_load_config("ok", str(tmp_path)))

    response = asyncio.run(_run())
    assert "warning" not in response
    assert response["status"] == "loaded"
