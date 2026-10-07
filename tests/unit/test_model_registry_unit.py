import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
from src.server.model_registry import ModelRecord, ModelRegistry, create_model_instance
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelStatus,
    ModelType,
)


def _sample_load_config(name: str = "mock-model") -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path="/models/mock",
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )


def test_register_load_sets_status_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    async def _noop_unload(*_args, **_kwargs):
        return None

    dummy_model = SimpleNamespace(unload_model=_noop_unload)

    async def fake_create(config):  # type: ignore[override]
        assert config is load_config
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        model_id = await registry.register_load(load_config)
        status = await registry.status()
        return model_id, status

    model_id, status = asyncio.run(_run())

    assert model_id
    assert status["total_loaded_models"] == 1
    entry = status["models"][0]
    assert entry["model_name"] == load_config.model_name
    assert entry["status"] == ModelStatus.LOADED.value


def test_register_load_duplicate_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    async def _noop_unload(*_args, **_kwargs):
        return None

    dummy_model = SimpleNamespace(unload_model=_noop_unload)

    async def fake_create(config):  # type: ignore[override]
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        with pytest.raises(ValueError):
            await registry.register_load(load_config)

    asyncio.run(_run())


def test_register_unload_invokes_model_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    unload_calls = []

    class DummyModel:
        async def unload_model(self, reg, name):
            unload_calls.append((reg, name))

    async def fake_create(config):  # type: ignore[override]
        return DummyModel()

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        result = await registry.register_unload(load_config.model_name)
        assert result is True
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        status = await registry.status()
        return status

    status = asyncio.run(_run())

    assert unload_calls
    assert unload_calls[0][1] == load_config.model_name
    assert status["total_loaded_models"] == 0


def test_register_unload_unknown_model_returns_false() -> None:
    registry = ModelRegistry()

    async def _run():
        return await registry.register_unload("missing-model")

    result = asyncio.run(_run())
    assert result is False


def test_create_model_instance_rejects_unknown_combination() -> None:
    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="unsupported",
        model_type=ModelType.VLM,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    async def _run():
        with pytest.raises(ValueError) as exc:
            await create_model_instance(load_config)
        return str(exc.value)

    message = asyncio.run(_run())
    assert "not supported" in message


def test_model_class_registry_includes_qwen3_asr() -> None:
    key = (EngineType.OPENVINO, ModelType.QWEN3_ASR)
    assert registry_module.MODEL_CLASS_REGISTRY[key] == "src.engine.openvino.qwen3_asr.qwen3_asr.OVQwen3ASR"


# --- opt-in context-window advertisement threaded into the registry ---------
#
# Advertisement is opt-in: register_load only resolves a context window when the
# operator set load_config.context_window; an unset value stays None (nothing is
# even read from config.json). A pinned value above the real max_position_
# embeddings is still advertised as-is, and stashed as a loud warning that the
# load endpoint relays to the CLI (and the server logs).


def _write_model_dir(tmp_path: Path, name: str, payload: dict | None) -> str:
    """Create a fake model directory; only writes config.json when ``payload`` is set."""
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
    captured: dict | None = None,
) -> dict:
    """Register a load config with a faked factory; return the registry status.

    When ``captured`` is given it receives the loader the engine actually gets,
    proving the resolved window is advertised-only (never stamped onto it).
    """

    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        if captured is not None:
            captured["config"] = config
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        return await registry.status()

    return asyncio.run(_run())


def test_register_load_unset_stays_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Opt-out: even with a window in config.json, an unset opt-in leaves the
    # advertised value None -- nothing is read or advertised.
    model_path = _write_model_dir(tmp_path, "m", {"max_position_embeddings": 131072})
    status = _register(monkeypatch, ModelRegistry(), _load_config("m", model_path))
    assert status["models"][0]["context_window"] is None


def test_register_load_explicit_int_advertised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An explicit int is advertised as-is, not the config.json value.
    model_path = _write_model_dir(tmp_path, "m", {"max_position_embeddings": 40960})
    status = _register(monkeypatch, ModelRegistry(), _load_config("m", model_path, 20480))
    assert status["models"][0]["context_window"] == 20480


def test_register_load_auto_discovers_flat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "auto" discovers the flat top-level max_position_embeddings.
    model_path = _write_model_dir(tmp_path, "m", {"max_position_embeddings": 40960})
    status = _register(monkeypatch, ModelRegistry(), _load_config("m", model_path, "auto"))
    assert status["models"][0]["context_window"] == 40960


def test_register_load_auto_discovers_nested_text_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The VLM symptom: the window is nested under text_config (with a decoy
    # top-level sliding_window). "auto" discovers it, and the engine-facing loader
    # is left untouched (the resolved value is advertised-only, never stamped on).
    model_path = _write_model_dir(
        tmp_path,
        "m",
        {
            "model_type": "qwen2_5_vl",
            "sliding_window": 512,
            "text_config": {"max_position_embeddings": 131072},
            "vision_config": {"model_type": "qwen2_5_vl"},
        },
    )
    captured: dict = {}
    load_config = _load_config("m", model_path, "auto")
    registry = ModelRegistry()
    status = _register(monkeypatch, registry, load_config, captured)
    assert status["models"][0]["model_name"] == "m"
    assert status["models"][0]["context_window"] == 131072
    # The engine receives the original, un-stamped ("auto") loader.
    assert captured["config"] is load_config
    assert captured["config"].context_window == "auto"


def test_registered_models_view_only_advertised_value_not_warning() -> None:
    # The public view carries context_window (the advertised value) but never the
    # internal warning, so the warning can't leak to /v1/models.
    record = ModelRecord(
        model_name="m",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        context_window=1234,
        context_window_warning="some warning",
    )
    view = record.registered_models()
    assert view["context_window"] == 1234
    assert "context_window_warning" not in view
    # An opt-out (default) record advertises nothing and carries no warning.
    assert ModelRecord().registered_models()["context_window"] is None


# --- a pinned value above the real limit warns loudly ---------------------

def test_register_load_pinned_over_warns_and_still_advertises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model_path = _write_model_dir(tmp_path, "over", {"max_position_embeddings": 32768})
    registry = ModelRegistry()
    with caplog.at_level(logging.WARNING, logger=registry_module.logger.name):
        status = _register(monkeypatch, registry, _load_config("over", model_path, 131072))

    # The over-large value is what is advertised (the warning is about it)...
    assert status["models"][0]["context_window"] == 131072
    # ... stashed on the record for the load endpoint / CLI to relay...
    warning = registry.get_context_window_warning("over")
    assert warning is not None
    assert "131072" in warning and "32768" in warning
    # ... and logged loudly server-side (the "display" on `openarc serve`).
    assert any(
        "exceeds the model's real context" in log.getMessage() for log in caplog.records
    )


def test_register_load_pinned_at_limit_no_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = _write_model_dir(tmp_path, "ok", {"max_position_embeddings": 32768})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("ok", model_path, 32768))
    assert registry.get_context_window_warning("ok") is None


def test_register_load_auto_and_unset_never_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = ModelRegistry()
    # "auto" resolves *to* the real limit, so it can never exceed it.
    auto_path = _write_model_dir(tmp_path, "auto", {"max_position_embeddings": 32768})
    _register(monkeypatch, registry, _load_config("auto", auto_path, "auto"))
    assert registry.get_context_window_warning("auto") is None
    # Unset advertises nothing, so there is nothing to warn about either.
    unset_path = _write_model_dir(tmp_path, "unset", {"max_position_embeddings": 32768})
    _register(monkeypatch, registry, _load_config("unset", unset_path))
    assert registry.get_context_window_warning("unset") is None


def test_get_context_window_warning_unknown_model_returns_none() -> None:
    assert ModelRegistry().get_context_window_warning("does-not-exist") is None
