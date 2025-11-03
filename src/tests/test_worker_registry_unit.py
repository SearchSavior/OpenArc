import asyncio

import pytest  # type: ignore[import]

import src.server.worker_registry as worker_module
from src.server.model_registry import ModelRecord, ModelRegistry, ModelType
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
def worker_registry(monkeypatch: pytest.MonkeyPatch) -> worker_module.WorkerRegistry:
    class DummyLLM:  # noqa: D401
        pass

    class DummyVLM:  # noqa: D401
        pass

    class DummyWhisper:  # noqa: D401
        pass

    class DummyKokoro:  # noqa: D401
        pass

    class DummyEmb:  # noqa: D401
        pass

    class DummyRR:  # noqa: D401
        pass

    monkeypatch.setattr(worker_module, "OVGenAI_LLM", DummyLLM)
    monkeypatch.setattr(worker_module, "OVGenAI_VLM", DummyVLM)
    monkeypatch.setattr(worker_module, "OVGenAI_Whisper", DummyWhisper)
    monkeypatch.setattr(worker_module, "OV_Kokoro", DummyKokoro)
    monkeypatch.setattr(worker_module, "Optimum_EMB", DummyEmb)
    monkeypatch.setattr(worker_module, "Optimum_RR", DummyRR)

    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_llm",
        _make_worker("llm-full", {"tokens": 3}, supports_stream=True),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_vlm",
        _make_worker("vlm-full", {"tokens": 2}, supports_stream=True),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_whisper",
        _make_worker("whisper-text", {"words": 2}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_kokoro",
        _make_worker("audio-base64", {"chunks_processed": 1}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_emb",
        _make_worker([[0.1, 0.2]], {"dim": 2}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_rr",
        _make_worker([{"doc": "A", "score": 0.9}], {"total": 1}),
    )

    model_registry = ModelRegistry()
    return worker_module.WorkerRegistry(model_registry)


def _make_record(model_type: ModelType, model_name: str) -> ModelRecord:
    engine_map = {
        ModelType.LLM: "ov_genai",
        ModelType.VLM: "ov_genai",
        ModelType.WHISPER: "ov_genai",
        ModelType.KOKORO: "openvino",
        ModelType.EMB: "ov_optimum",
        ModelType.RERANK: "ov_optimum",
    }
    instance_factory = {
        ModelType.LLM: worker_module.OVGenAI_LLM,
        ModelType.VLM: worker_module.OVGenAI_VLM,
        ModelType.WHISPER: worker_module.OVGenAI_Whisper,
        ModelType.KOKORO: worker_module.OV_Kokoro,
        ModelType.EMB: worker_module.Optimum_EMB,
        ModelType.RERANK: worker_module.Optimum_RR,
    }
    record = ModelRecord(
        model_path="/models/mock",
        model_name=model_name,
        model_type=model_type,
        engine=engine_map[model_type],
        device="CPU",
    )
    record.model_instance = instance_factory[model_type]()  # type: ignore[call-arg]
    return record


async def _load_and_call(worker_registry, record, coro):
    await worker_registry._on_model_loaded(record)
    await asyncio.sleep(0)
    result = await coro
    await worker_registry._on_model_unloaded(record)
    await asyncio.sleep(0)
    return result


def test_llm_generate(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.LLM, "llm-model")
    config = OVGenAI_GenConfig(prompt="hello")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate("llm-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "llm-full", "metrics": {"tokens": 3}}


def test_llm_stream(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.LLM, "llm-stream")
    config = OVGenAI_GenConfig(prompt="hi", stream=True)

    async def _run():
        await worker_registry._on_model_loaded(record)
        await asyncio.sleep(0)
        outputs = []
        async for item in worker_registry.stream_generate("llm-stream", config):
            outputs.append(item)
        await worker_registry._on_model_unloaded(record)
        await asyncio.sleep(0)
        return outputs

    outputs = asyncio.run(_run())
    assert outputs == ["llm-full", {"metrics": {"tokens": 3}}]


def test_vlm_generate(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.VLM, "vlm-model")
    config = OVGenAI_GenConfig(prompt="describe image")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate("vlm-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "vlm-full", "metrics": {"tokens": 2}}


def test_whisper_transcribe(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.WHISPER, "whisper-model")
    config = OVGenAI_WhisperGenConfig(audio_base64="AAA")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.transcribe_whisper("whisper-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "whisper-text", "metrics": {"words": 2}}


def test_kokoro_generate_speech(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.KOKORO, "kokoro-model")
    config = OV_KokoroGenConfig(
        kokoro_message="Hello",
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=50,
        response_format="wav",
    )

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate_speech_kokoro("kokoro-model", config))

    result = asyncio.run(_run())
    assert result == {"audio_base64": "audio-base64", "metrics": {"chunks_processed": 1}}


def test_embed(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.EMB, "emb-model")
    config = PreTrainedTokenizerConfig(text=["embed me"])

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.embed("emb-model", config))

    result = asyncio.run(_run())
    assert result == {"data": [[0.1, 0.2]], "metrics": {"dim": 2}}


def test_rerank(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.RERANK, "rerank-model")
    config = RerankerConfig(query="Paris", documents=["Paris", "Berlin"])

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.rerank("rerank-model", config))

    result = asyncio.run(_run())
    assert result == {"data": [{"doc": "A", "score": 0.9}], "metrics": {"total": 1}}


def test_missing_model_queue(worker_registry: worker_module.WorkerRegistry) -> None:
    with pytest.raises(ValueError):
        worker_registry._get_model_queue("missing")

