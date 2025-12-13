import asyncio

import pytest  # type: ignore[import]

import src.server.model_registry as model_registry_module
import src.server.worker_registry as worker_module
from src.server.model_registry import ModelRegistry
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.openvino import KokoroLanguage, KokoroVoice, OV_KokoroGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig


def _make_worker(response_value, metrics_value, supports_stream: bool = False):
    async def _worker(model_name, model_queue, model_instance, registry):
        while True:
            packet = await model_queue.get()
            if packet is None:
                break
            if getattr(packet.gen_config, "stream", False) and supports_stream and packet.stream_queue is not None:
                await packet.stream_queue.put(response_value)
                if metrics_value is not None:
                    await packet.stream_queue.put({"metrics": metrics_value})
                await packet.stream_queue.put(None)
            packet.response = response_value
            packet.metrics = metrics_value
            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(packet)
            model_queue.task_done()
    return _worker


@pytest.fixture
def worker_system(monkeypatch: pytest.MonkeyPatch):
    class DummyLLM:
        async def unload_model(self, registry, model_name):
            return True

    class DummyVLM:
        async def unload_model(self, registry, model_name):
            return True

    class DummyWhisper:
        async def unload_model(self, registry, model_name):
            return True

    class DummyKokoro:
        async def unload_model(self, registry, model_name):
            return True

    class DummyEmb:
        async def unload_model(self, registry, model_name):
            return True

    class DummyRR:
        async def unload_model(self, registry, model_name):
            return True

    monkeypatch.setattr(worker_module, "OVGenAI_LLM", DummyLLM)
    monkeypatch.setattr(worker_module, "OVGenAI_VLM", DummyVLM)
    monkeypatch.setattr(worker_module, "OVGenAI_Whisper", DummyWhisper)
    monkeypatch.setattr(worker_module, "OV_Kokoro", DummyKokoro)
    monkeypatch.setattr(worker_module, "Optimum_EMB", DummyEmb)
    monkeypatch.setattr(worker_module, "Optimum_RR", DummyRR)

    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_llm", _make_worker("llm-full", {"tokens": 3}, True))
    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_vlm", _make_worker("vlm-full", {"tokens": 2}, True))
    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_whisper", _make_worker("whisper-text", {"words": 2}))
    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_kokoro", _make_worker("audio-base64", {"chunks_processed": 1}))
    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_emb", _make_worker([[0.1, 0.2]], {"dim": 2}))
    monkeypatch.setattr(worker_module.QueueWorker, "queue_worker_rr", _make_worker([{"doc": "A", "score": 0.9}], {"total": 1}))

    type_map = {
        ModelType.LLM: DummyLLM,
        ModelType.VLM: DummyVLM,
        ModelType.WHISPER: DummyWhisper,
        ModelType.KOKORO: DummyKokoro,
        ModelType.EMB: DummyEmb,
        ModelType.RERANK: DummyRR,
    }

    async def fake_create_model_instance(config: ModelLoadConfig):  # type: ignore[override]
        cls = type_map[config.model_type]
        return cls()

    monkeypatch.setattr(model_registry_module, "create_model_instance", fake_create_model_instance)

    model_registry = ModelRegistry()
    worker_registry = worker_module.WorkerRegistry(model_registry)
    return model_registry, worker_registry


async def _load_do_unload(model_registry, worker_registry, load_config, coro):
    await model_registry.register_load(load_config)
    await asyncio.sleep(0)
    result = await coro
    await model_registry.register_unload(load_config.model_name)
    await asyncio.sleep(0)
    return result


def test_worker_registry_generate_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-llm",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    config = OVGenAI_GenConfig(prompt="hello")

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.generate("integration-llm", config))

    result = asyncio.run(_run())
    assert result == {"text": "llm-full", "metrics": {"tokens": 3}}


def test_worker_registry_stream_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-stream",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    config = OVGenAI_GenConfig(prompt="stream", stream=True)

    async def _run():
        await model_registry.register_load(load_config)
        await asyncio.sleep(0)
        items = []
        async for item in worker_registry.stream_generate("integration-stream", config):
            items.append(item)
        await model_registry.register_unload("integration-stream")
        await asyncio.sleep(0)
        return items

    outputs = asyncio.run(_run())
    assert outputs == ["llm-full", {"metrics": {"tokens": 3}}]


def test_worker_registry_vlm_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-vlm",
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    config = OVGenAI_GenConfig(prompt="describe")

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.generate("integration-vlm", config))

    result = asyncio.run(_run())
    assert result == {"text": "vlm-full", "metrics": {"tokens": 2}}


def test_worker_registry_whisper_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-whisper",
        model_type=ModelType.WHISPER,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    config = OVGenAI_WhisperGenConfig(audio_base64="AAA")

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.transcribe_whisper("integration-whisper", config))

    result = asyncio.run(_run())
    assert result == {"text": "whisper-text", "metrics": {"words": 2}}


def test_worker_registry_kokoro_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-kokoro",
        model_type=ModelType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",
        runtime_config={},
    )

    config = OV_KokoroGenConfig(
        kokoro_message="Hello",
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=100,
        response_format="wav",
    )

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.generate_speech_kokoro("integration-kokoro", config))

    result = asyncio.run(_run())
    assert result == {"audio_base64": "audio-base64", "metrics": {"chunks_processed": 1}}


def test_worker_registry_embed_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-emb",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    config = PreTrainedTokenizerConfig(text=["embed"])

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.embed("integration-emb", config))

    result = asyncio.run(_run())
    assert result == {"data": [[0.1, 0.2]], "metrics": {"dim": 2}}


def test_worker_registry_rerank_flow(worker_system) -> None:
    model_registry, worker_registry = worker_system

    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="integration-rr",
        model_type=ModelType.RERANK,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    config = RerankerConfig(query="Paris", documents=["Paris", "Berlin"])

    async def _run():
        return await _load_do_unload(model_registry, worker_registry, load_config, worker_registry.rerank("integration-rr", config))

    result = asyncio.run(_run())
    assert result == {"data": [{"doc": "A", "score": 0.9}], "metrics": {"total": 1}}

