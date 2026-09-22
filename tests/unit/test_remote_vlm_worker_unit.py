"""
End-to-end tests for the out-of-process VLM worker (stage 1):
supervisor + protocol + RemoteOVGenAI_VLM facade, with the worker's model
replaced by a stub (OPENARC_WORKER_STUB=1) so no OpenVINO, GPU, or model
files are needed. Each test spawns a real child process and exercises the
full pipe: spawn -> LOAD -> GENERATE/ITEM/DONE -> UNLOAD, plus the failure
paths (recoverable error, FATAL+respawn, native-crash+respawn, respawn
budget exhaustion, load failure).
"""

import asyncio

import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto
from src.engine.worker.worker_client import RemoteOVGenAI_VLM
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType


def _load_config(tmp_path, name: str = "stub-vlm") -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(tmp_path),
        model_name=name,
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
    )


def _gen_config(prompt: str, stream: bool = False, request_id: str = None) -> OVGenAI_GenConfig:
    return OVGenAI_GenConfig(prompt=prompt, stream=stream, request_id=request_id)


async def _wait_for_state(facade: RemoteOVGenAI_VLM, state: str, timeout: float = 30.0) -> None:
    async def _poll():
        while facade.status()["state"] != state:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


async def _wait_for_respawn(facade: RemoteOVGenAI_VLM, first_pid: int, timeout: float = 30.0) -> None:
    """Wait until the supervisor has detected the death and the respawned
    worker is ready again (a fresh PID proves a fresh process/Core)."""

    async def _poll():
        while facade.worker_pid == first_pid or facade.status()["state"] != "ready":
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


async def _drain(facade: RemoteOVGenAI_VLM, gen_config: OVGenAI_GenConfig) -> list:
    return [item async for item in facade.generate_type(gen_config)]


def test_load_generate_and_unload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        assert facade.status()["state"] == "ready"
        assert facade.worker_pid is not None

        # Non-streaming contract: metrics dict first, then the full text.
        items = await _drain(facade, _gen_config("hello world"))
        assert isinstance(items[0], dict)
        assert items[0]["new_token"] == 2
        assert items[1] == "hello world"

        await facade._supervisor.unload()
        assert facade.status()["state"] == "closed"

    asyncio.run(_run())


def test_streaming_yields_chunks_then_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    monkeypatch.setenv("OPENARC_WORKER_STUB_TOKEN_DELAY", "0.01")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        items = await _drain(facade, _gen_config("a b c", stream=True))
        assert [i for i in items if isinstance(i, str)] == ["a", "b", "c"]
        assert items[-1] == {"new_token": 3, "stream": True}
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_recoverable_error_fails_request_but_worker_survives(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)

        with pytest.raises(proto.RemoteWorkerError) as exc:
            await _drain(facade, _gen_config("RAISE"))
        assert "stub recoverable error" in str(exc.value)
        assert exc.value.original_type == "RuntimeError"

        # The worker process is still alive and serving.
        assert facade.status()["state"] == "ready"
        items = await _drain(facade, _gen_config("still alive"))
        assert items[-1] == "still alive"
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_fatal_error_respawns_worker(tmp_path, monkeypatch) -> None:
    """A non-recoverable (CL_*) error takes the process down; the supervisor
    respawns a fresh one and the model serves again without a registry reload."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        first_pid = facade.worker_pid

        with pytest.raises(proto.RemoteWorkerError) as exc:
            await _drain(facade, _gen_config("FATAL please"))
        assert "CL_OUT_OF_RESOURCES" in str(exc.value)

        # The supervisor uses its respawn budget to bring the model back.
        await _wait_for_respawn(facade, first_pid)
        assert facade.status()["respawns"] == 1

        items = await _drain(facade, _gen_config("back up"))
        assert items[-1] == "back up"
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_unexpected_crash_respawns_worker(tmp_path, monkeypatch) -> None:
    """A native-style crash (process dies without a FATAL message) is also
    recovered by a respawn."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        first_pid = facade.worker_pid

        with pytest.raises(proto.RemoteWorkerDeadError):
            await _drain(facade, _gen_config("CRASH"))

        await _wait_for_respawn(facade, first_pid)
        items = await _drain(facade, _gen_config("still here"))
        assert items[-1] == "still here"
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_cancel_stops_stream_early(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    monkeypatch.setenv("OPENARC_WORKER_STUB_TOKEN_DELAY", "0.05")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        got: list = []

        async def consume():
            async for item in facade.generate_type(
                _gen_config("SLOW one two three four five", stream=True, request_id="req-1")
            ):
                got.append(item)

        task = asyncio.create_task(consume())
        # Let a couple of tokens arrive, then cancel.
        for _ in range(500):
            if len([g for g in got if isinstance(g, str)]) >= 2:
                break
            await asyncio.sleep(0.01)

        cancelled = await facade.cancel("req-1")
        assert cancelled is True
        await task

        # The stream ended early: fewer than all five tokens, and a cancelled
        # stream yields no trailing metrics dict.
        text_chunks = [g for g in got if isinstance(g, str)]
        assert 2 <= len(text_chunks) < 5
        assert not any(isinstance(g, dict) for g in got)

        # Cancellation of an unknown request id reports False.
        assert await facade.cancel("no-such-request") is False
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_respawn_budget_exhaustion_unloads_model(tmp_path, monkeypatch) -> None:
    class _FakeRegistry:
        def __init__(self):
            self.unloaded: list = []

        async def register_unload(self, model_name: str, administrative: bool = False) -> bool:
            self.unloaded.append(model_name)
            return True

    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        registry = _FakeRegistry()
        facade = RemoteOVGenAI_VLM(
            config, registry=registry, max_respawns=0, load_timeout=30.0
        )
        await facade.load_model(config)

        with pytest.raises(proto.RemoteWorkerError):
            await _drain(facade, _gen_config("CRASH"))

        # Budget of 0: no respawn; the facade reports the death to the registry.
        await _wait_for_state(facade, "dead")
        for _ in range(500):
            if registry.unloaded:
                break
            await asyncio.sleep(0.01)
        assert registry.unloaded == [config.model_name]

        # And further requests fail fast.
        with pytest.raises(proto.RemoteWorkerDeadError):
            await _drain(facade, _gen_config("again"))
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_load_failure_fails_start_and_cleans_up(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    monkeypatch.setenv("OPENARC_WORKER_STUB_FAIL_LOAD", "1")
    config = _load_config(tmp_path)

    async def _run():
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        with pytest.raises(proto.RemoteWorkerLoadError) as exc:
            await facade.load_model(config)
        assert "stub load failure" in str(exc.value)
        # The failed episode ends in DEAD and the child process is reaped.
        assert facade.status()["state"] == "dead"
        await facade._supervisor.unload()
        assert facade.status()["state"] == "closed"

    asyncio.run(_run())


def test_registry_end_to_end_with_remote_vlm(tmp_path, monkeypatch) -> None:
    """Full path: ModelRegistry.register_load -> RemoteOVGenAI_VLM facade ->
    worker process -> WorkerRegistry.generate (packets + queue worker), plus
    the policy that a worker death must NOT trigger a registry unload (the
    supervisor owns recovery)."""
    from src.server.model_registry import ModelRegistry
    from src.server.worker_registry import WorkerRegistry

    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run():
        registry = ModelRegistry()
        workers = WorkerRegistry(registry)

        model_id = await registry.register_load(config)
        record = None
        async with registry._lock:
            for rec in registry._models.values():
                if rec.model_id == model_id:
                    record = rec
        assert record is not None
        assert isinstance(record.model_instance, RemoteOVGenAI_VLM)

        # Non-streaming generate through the full packet/worker pipeline.
        result = await workers.generate(config.model_name, _gen_config("hello world"))
        assert result["text"] == "hello world"
        first_pid = record.model_instance.worker_pid

        # A fatal error fails the request but must NOT unload the model: the
        # supervisor respawns and the model keeps serving.
        with pytest.raises(proto.RemoteWorkerError):
            await workers.generate(config.model_name, _gen_config("FATAL please"))
        await _wait_for_respawn(record.model_instance, first_pid)
        async with registry._lock:
            assert any(r.model_name == config.model_name for r in registry._models.values())

        result = await workers.generate(config.model_name, _gen_config("still serving"))
        assert result["text"] == "still serving"

        # Streaming works too.
        streamed = []
        async for item in workers.stream_generate(config.model_name, _gen_config("a b c", stream=True)):
            streamed.append(item)
        assert [i for i in streamed if isinstance(i, str)] == ["a", "b", "c"]

        # Unload terminates the worker process.
        assert await registry.register_unload(config.model_name) is True
        for _ in range(500):
            if record.model_instance.status()["state"] == "closed":
                break
            await asyncio.sleep(0.01)
        assert record.model_instance.status()["state"] == "closed"

    asyncio.run(_run())
