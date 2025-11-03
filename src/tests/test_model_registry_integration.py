import asyncio

import pytest  # type: ignore[import]

import src.server.model_registry as model_registry_module
from src.server.model_registry import EngineType, ModelLoadConfig, ModelRegistry, ModelType


class _DummyModel:
    def __init__(self):
        self.loaded = False
        self.unloaded = False

    def load_model(self, load_config: ModelLoadConfig) -> None:
        self.loaded = True

    async def unload_model(self, registry, model_name: str) -> bool:
        self.unloaded = True
        return True


@pytest.fixture
def patched_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_model_instance(load_config: ModelLoadConfig):  # type: ignore[override]
        model = _DummyModel()
        model.load_model(load_config)
        return model

    monkeypatch.setattr(model_registry_module, "create_model_instance", fake_create_model_instance)


def test_register_load_and_unload_flow(patched_model_factory) -> None:
    registry = ModelRegistry()
    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    async def _run():
        model_id = await registry.register_load(load_config)
        status_loaded = await registry.status()
        await registry.register_unload(load_config.model_name)
        await asyncio.sleep(0)
        status_after = await registry.status()
        return model_id, status_loaded, status_after

    model_id, status_loaded, status_after = asyncio.run(_run())

    assert model_id
    assert status_loaded["total_loaded_models"] == 1
    assert status_loaded["models"][0]["status"] == "loaded"
    assert status_after["total_loaded_models"] == 0


def test_register_load_failure_marks_status(patched_model_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-fail",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    async def failing_create(load_config):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(model_registry_module, "create_model_instance", failing_create)

    async def _run():
        with pytest.raises(RuntimeError):
            await registry.register_load(load_config)
        status = await registry.status()
        return status

    status = asyncio.run(_run())
    assert status["total_loaded_models"] == 1
    assert status["models"][0]["status"] == "failed"


def test_unload_unknown_model_returns_false(patched_model_factory) -> None:
    registry = ModelRegistry()

    async def _run():
        return await registry.register_unload("missing")

    result = asyncio.run(_run())
    assert result is False

