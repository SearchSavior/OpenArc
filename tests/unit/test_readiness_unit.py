import asyncio
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
from src.server.model_registry import ModelRecord, ModelRegistry
from src.server.models.registration import (
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


def _patch_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(_config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)


# --- readiness() logic -------------------------------------------------------


def test_readiness_empty_registry_is_not_ready() -> None:
    """No models expected at all means the server is not ready."""
    registry = ModelRegistry()

    result = asyncio.run(registry.readiness())

    assert result == {
        "ready": False,
        "expected_models": [],
        "missing_models": [],
    }


def test_readiness_ready_when_expected_model_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        return await registry.readiness()

    result = asyncio.run(_run())

    assert result["ready"] is True
    assert result["expected_models"] == ["a"]
    assert result["missing_models"] == []


def test_readiness_not_ready_when_an_expected_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model expected but only present as a non-LOADED record is missing."""
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        await registry.register_load(_sample_load_config("b"))
        # Simulate 'b' degrading without leaving the registry (e.g. mid-reload).
        async with registry._lock:
            for record in registry._models.values():
                if record.model_name == "b":
                    record.status = ModelStatus.FAILED
        return await registry.readiness()

    result = asyncio.run(_run())

    assert result["ready"] is False
    assert result["expected_models"] == ["a", "b"]
    assert result["missing_models"] == ["b"]


# --- expected-set lifecycle via (un)load ------------------------------------


def test_error_unload_keeps_model_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-administrative unload (e.g. worker error) leaves the model expected."""
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        # Default administrative=False mirrors the worker error path.
        await registry.register_unload("a")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return await registry.readiness()

    result = asyncio.run(_run())

    # Model dropped out of the registry but is still required -> not ready.
    assert result["ready"] is False
    assert result["expected_models"] == ["a"]
    assert result["missing_models"] == ["a"]


def test_administrative_unload_drops_model_from_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        await registry.register_load(_sample_load_config("b"))
        await registry.register_unload("a", administrative=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return await registry.readiness()

    result = asyncio.run(_run())

    # 'a' is no longer required; 'b' is still loaded -> ready.
    assert result["ready"] is True
    assert result["expected_models"] == ["b"]
    assert result["missing_models"] == []


def test_administrative_unload_of_last_model_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        await registry.register_unload("a", administrative=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return await registry.readiness()

    result = asyncio.run(_run())

    assert result["ready"] is False
    assert result["expected_models"] == []


def test_administrative_unload_clears_expectation_for_absent_model() -> None:
    """An operator can clear a stale expectation even after the record is gone."""
    registry = ModelRegistry()

    async def _run():
        # Model errored out earlier: still expected, but no longer in _models.
        registry._expected_models.add("ghost")
        before = await registry.readiness()
        # Explicit unload returns False (not present) but must clear expectation.
        result = await registry.register_unload("ghost", administrative=True)
        after = await registry.readiness()
        return before, result, after

    before, result, after = asyncio.run(_run())

    assert before["missing_models"] == ["ghost"]
    assert result is False
    assert after["expected_models"] == []


def test_reload_after_administrative_unload_restores_expectation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory(monkeypatch)
    registry = ModelRegistry()

    async def _run():
        await registry.register_load(_sample_load_config("a"))
        await registry.register_unload("a", administrative=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await registry.register_load(_sample_load_config("a"))
        return await registry.readiness()

    result = asyncio.run(_run())

    assert result["ready"] is True
    assert result["expected_models"] == ["a"]


# --- /readyz endpoint --------------------------------------------------------


def _make_loaded_record(name: str) -> ModelRecord:
    return ModelRecord(
        model_name=name,
        model_type="llm",
        engine="ov_genai",
        device="CPU",
        status=ModelStatus.LOADED,
    )


def test_readyz_endpoint_status_codes() -> None:
    from starlette.testclient import TestClient

    from src.server import deps
    from src.server.main import app

    registry = deps._registry
    # Snapshot and isolate the shared singleton's state for this test.
    saved_models = dict(registry._models)
    saved_expected = set(registry._expected_models)
    registry._models.clear()
    registry._expected_models.clear()
    try:
        with TestClient(app) as client:
            # No models -> not ready.
            resp = client.get("/readyz")
            assert resp.status_code == 503
            assert resp.json()["ready"] is False

            # One loaded + expected model -> ready.
            record = _make_loaded_record("a")
            registry._models[record.model_id] = record
            registry._expected_models.add("a")

            resp = client.get("/readyz")
            assert resp.status_code == 200
            body = resp.json()
            assert body["ready"] is True
            assert body["expected_models"] == ["a"]

            # Expected model drops out (error unload) -> not ready again.
            registry._models.clear()
            resp = client.get("/readyz")
            assert resp.status_code == 503
            assert resp.json()["missing_models"] == ["a"]
    finally:
        registry._models.clear()
        registry._models.update(saved_models)
        registry._expected_models.clear()
        registry._expected_models.update(saved_expected)
