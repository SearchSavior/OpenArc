import asyncio
import json
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


# --- context-window discovery threaded into the registry --------------------

def _write_model_dir(tmp_path: Path, name: str, payload: dict) -> str:
    """Create a fake model directory (with a config.json) and return its path."""
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(model_dir)


def test_register_load_derives_context_window_from_config_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no explicit value, the window is discovered from config.json (first present key)."""
    model_path = _write_model_dir(
        tmp_path, "ctx-model", {"n_ctx": 8192, "max_position_embeddings": 40960}
    )
    load_config = ModelLoadConfig(
        model_path=model_path,
        model_name="ctx-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )
    registry = ModelRegistry()

    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        return await registry.status()

    status = asyncio.run(_run())
    assert status["models"][0]["context_window"] == 40960


def test_register_load_explicit_context_window_overrides_derivation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit positive context_window on the load config wins over config.json."""
    model_path = _write_model_dir(tmp_path, "ctx-override", {"n_ctx": 40960})
    load_config = ModelLoadConfig(
        model_path=model_path,
        model_name="ctx-override",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
        context_window=20480,
    )
    registry = ModelRegistry()

    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        return await registry.status()

    status = asyncio.run(_run())
    assert status["models"][0]["context_window"] == 20480


def test_register_load_context_window_none_without_config_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No config.json and no explicit value -> None (clients fall back to defaults)."""
    model_dir = tmp_path / "no-config"
    model_dir.mkdir()
    load_config = ModelLoadConfig(
        model_path=str(model_dir),
        model_name="no-config",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )
    registry = ModelRegistry()

    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        return await registry.status()

    status = asyncio.run(_run())
    assert status["models"][0]["context_window"] is None


def test_registered_models_exposes_context_window_key() -> None:
    """The public view always carries the key (None when unknown)."""
    record = ModelRecord(
        model_name="m",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        context_window=1234,
    )
    assert record.registered_models()["context_window"] == 1234
    assert ModelRecord().registered_models()["context_window"] is None

