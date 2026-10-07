"""Unit tests for the /v1/models context-window advertisement.

Exercises the actual ``openai_list_models`` handler end to end: a model is
registered in a real ``ModelRegistry`` (with a fake factory, so no heavy engine
is loaded), then the OpenAI-compatible ``/v1/models`` response is built.

Advertisement is opt-in, so these pin the resulting behavior:

* ``context_window`` unset -> nothing is advertised (the opt-out default), so the
  entry has no ``context_window`` / ``meta`` keys at all -- no default is even
  read from config.json.
* ``context_window: "auto"`` -> the model's ``max_position_embeddings`` is
  advertised (in ``context_window`` and ``meta.n_ctx``); the decoy keys other
  exporters use are not.
* a positive integer -> that exact value is advertised (used as-is).
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
import src.server.routes.openai as openai_module
from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelType,
)


def _write_model_dir(tmp_path: Path, name: str, payload: dict | None) -> str:
    """Create a fake model directory; only writes config.json when ``payload`` is set.

    Also drops the _model.bin/_model.xml the model-path check expects, so a
    later real load is not what we are testing here (the factory is faked).
    """
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True)
    if payload is not None:
        (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(model_dir)


def _load_config(
    name: str, model_path: str, context_window: int | str | None = None
) -> ModelLoadConfig:
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


def _register(
    monkeypatch: pytest.MonkeyPatch,
    registry: ModelRegistry,
    load_config: ModelLoadConfig,
) -> None:
    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)

    asyncio.run(_run())


def _list(monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry) -> list:
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())
    return {entry["id"]: entry for entry in response["data"]}


# --- opt-out: the default advertises nothing ---------------------------

def test_unset_advertises_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # config.json *does* carry a window, but nothing is advertised because the
    # operator never opted in -- a default must not be read or advertised.
    model_path = _write_model_dir(tmp_path, "unset", {"max_position_embeddings": 131072})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("unset", model_path))

    entry = _list(monkeypatch, registry)["unset"]
    assert entry["object"] == "model"
    assert entry["owned_by"] == "OpenArc"
    assert "context_window" not in entry
    assert "meta" not in entry


def test_auto_with_no_config_advertises_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "auto" but config.json has no max_position_embeddings -> still nothing.
    model_path = _write_model_dir(tmp_path, "auto-none", None)
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("auto-none", model_path, "auto"))

    entry = _list(monkeypatch, registry)["auto-none"]
    assert "context_window" not in entry
    assert "meta" not in entry


# --- opt-in via "auto": advertises only max_position_embeddings --------

def test_auto_advertises_max_position_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # max_position_embeddings wins; the decoy n_ctx/sliding_window are ignored.
    model_path = _write_model_dir(
        tmp_path,
        "auto",
        {
            "max_position_embeddings": 131072,
            "n_ctx": 32768,
            "sliding_window": 512,
        },
    )
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("auto", model_path, "auto"))

    entry = _list(monkeypatch, registry)["auto"]
    assert entry["context_window"] == 131072
    assert entry["meta"] == {"n_ctx": 131072}


def test_auto_discovers_nested_text_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The reported symptom: a multimodal config nests the window under text_config
    # with a decoy top-level sliding_window. "auto" surfaces the real window in
    # both fields clients read.
    model_path = _write_model_dir(
        tmp_path,
        "qwen25vl",
        {
            "model_type": "qwen2_5_vl",
            "sliding_window": 512,
            "text_config": {"max_position_embeddings": 32768},
            "vision_config": {"model_type": "qwen2_5_vl"},
        },
    )
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("qwen25vl", model_path, "auto"))

    entry = _list(monkeypatch, registry)["qwen25vl"]
    assert entry["context_window"] == 32768
    assert entry["meta"] == {"n_ctx": 32768}


# --- opt-in via an explicit integer: advertised as-is ------------------

def test_explicit_int_advertised_as_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pinned integer is advertised verbatim -- not the config.json value.
    model_path = _write_model_dir(tmp_path, "pinned", {"max_position_embeddings": 9999})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("pinned", model_path, 5000))

    entry = _list(monkeypatch, registry)["pinned"]
    assert entry["context_window"] == 5000
    assert entry["meta"] == {"n_ctx": 5000}


# --- the advertised value is what is warned about ----------------------

def test_explicit_over_advertises_the_larger_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pin above the real limit is what gets advertised (the warning, logged
    # server-side, is about this advertised value); the real limit is not emitted.
    model_path = _write_model_dir(tmp_path, "over", {"max_position_embeddings": 32768})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("over", model_path, 131072))

    entry = _list(monkeypatch, registry)["over"]
    assert entry["context_window"] == 131072
    assert entry["meta"] == {"n_ctx": 131072}


def test_mixed_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One model opts in via "auto"; another leaves it unset. Only the opted-in
    # one carries the fields.
    opted_in = _write_model_dir(tmp_path, "on", {"max_position_embeddings": 131072})
    opted_out = _write_model_dir(tmp_path, "off", {"max_position_embeddings": 6000})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("on", opted_in, "auto"))
    _register(monkeypatch, registry, _load_config("off", opted_out))

    entries = _list(monkeypatch, registry)
    assert entries["on"]["context_window"] == 131072
    assert entries["on"]["meta"] == {"n_ctx": 131072}
    assert "context_window" not in entries["off"]
    assert "meta" not in entries["off"]
