import asyncio
import threading
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest  # type: ignore[import]
from openvino_genai import GenerationStatus

import src.engine.ov_genai.continuous_batch_llm as cb_module
import src.engine.ov_genai.utils as utils_module
from src.engine.ov_genai.continuous_batch_llm import OVGenAI_ContinuousBatch
from src.engine.ov_genai.streamers import ChunkStreamer
from src.server.model_registry import MODEL_CLASS_REGISTRY
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig


MODEL_PATH = "some_fake_url/Qwen3-0.6B-fp16-ov"


class DummyMeanValue:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class DummyPerfMetrics:
    def get_load_time(self) -> int:
        return 5000

    def get_ttft(self) -> DummyMeanValue:
        return DummyMeanValue(250.0)

    def get_tpot(self) -> DummyMeanValue:
        return DummyMeanValue(7.5)

    def get_throughput(self) -> DummyMeanValue:
        return DummyMeanValue(12.34567)

    def get_generate_duration(self) -> DummyMeanValue:
        return DummyMeanValue(1000.0)

    def get_num_input_tokens(self) -> int:
        return 25

    def get_num_generated_tokens(self) -> int:
        return 10


class FakeOVTokenizer:
    """Stands in for the pipeline's openvino_genai Tokenizer.

    decode(1D) -> str, decode(2D) -> [str], mirroring the real API surface the
    streamers and the engine use.
    """

    def __init__(self, vocab: Optional[Dict[int, str]] = None) -> None:
        self.vocab = vocab or {}

    def decode(self, ids, skip_special_tokens: bool = True):
        if ids and isinstance(ids[0], list):
            return [self._decode_row(row) for row in ids]
        return self._decode_row(ids)

    def _decode_row(self, row) -> str:
        return "".join(self.vocab.get(i, f"<{i}>") for i in row)


class FakeGenerationOutput:
    def __init__(self, ids: List[int]) -> None:
        self.generated_ids = ids
        self.finish_reason = None
        self.score = 0.0


class FakeHandle:
    """Scripted GenerationHandle.

    script is a list of per-step token id batches; the string "IGNORED" as a
    step flips the handle to GenerationStatus.IGNORED (per-request KV OOM).
    When the script is exhausted and every token has been read, the handle
    reports FINISHED.
    """

    def __init__(self, request_id: int, script: list, perf_raises: bool = False) -> None:
        self.request_id = request_id
        self._script = list(script)
        self._pending: List[int] = []
        self._status = GenerationStatus.RUNNING
        self._perf_raises = perf_raises
        self.cancelled = False
        self.stopped = False

    def advance(self) -> None:
        if self._status != GenerationStatus.RUNNING:
            return
        if self._script:
            step = self._script.pop(0)
            if step == "IGNORED":
                self._status = GenerationStatus.IGNORED
                return
            self._pending.extend(step)

    def can_read(self) -> bool:
        return bool(self._pending)

    def read(self) -> Dict[int, FakeGenerationOutput]:
        ids, self._pending = self._pending, []
        return {0: FakeGenerationOutput(ids)}

    def get_status(self) -> GenerationStatus:
        if (
            self._status == GenerationStatus.RUNNING
            and not self._script
            and not self._pending
        ):
            self._status = GenerationStatus.FINISHED
        return self._status

    def cancel(self) -> None:
        self.cancelled = True
        if self._status == GenerationStatus.RUNNING:
            self._status = GenerationStatus.CANCEL

    def stop(self) -> None:
        self.stopped = True
        if self._status == GenerationStatus.RUNNING:
            self._status = GenerationStatus.STOP

    def get_perf_metrics(self) -> DummyPerfMetrics:
        if self._perf_raises:
            raise RuntimeError("perf metrics unavailable for this handle state")
        return DummyPerfMetrics()


class FakeCBPipeline:
    """Stands in for openvino_genai.ContinuousBatchingPipeline.

    scripts maps the internal (monotonic int) request id to a handle script.
    first_step_gate, when set, blocks the first step() call until the test
    releases it (the executor thread waits on a threading.Event).
    """

    def __init__(
        self,
        tokenizer: FakeOVTokenizer,
        scripts: Optional[Dict[int, list]] = None,
        first_step_gate: Optional[threading.Event] = None,
        perf_raises_ids: Optional[set] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._scripts = scripts or {}
        self._perf_raises_ids = perf_raises_ids or set()
        self._first_step_gate = first_step_gate
        self.handles: Dict[int, FakeHandle] = {}
        self.add_calls: list = []
        self.step_count = 0

    def add_request(self, request_id, *args) -> FakeHandle:
        self.add_calls.append((request_id,) + args)
        handle = FakeHandle(
            request_id,
            self._scripts.get(request_id, [[1]]),
            perf_raises=request_id in self._perf_raises_ids,
        )
        self.handles[request_id] = handle
        return handle

    def step(self) -> None:
        if self._first_step_gate is not None:
            gate, self._first_step_gate = self._first_step_gate, None
            gate.wait(timeout=10)
        self.step_count += 1
        for handle in list(self.handles.values()):
            handle.advance()

    def has_non_finished_requests(self) -> bool:
        return any(
            h.get_status() == GenerationStatus.RUNNING for h in self.handles.values()
        )

    def get_tokenizer(self) -> FakeOVTokenizer:
        return self._tokenizer

    def get_config(self):
        from openvino_genai import GenerationConfig

        return GenerationConfig()


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="test-cb-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI_CB,
        device="CPU",
        runtime_config={},
    )


@pytest.fixture
def vlm_load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="test-cb-vlm",
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI_CB,
        device="CPU",
        runtime_config={},
    )


def _make_engine(load_config: ModelLoadConfig, pipeline: FakeCBPipeline) -> OVGenAI_ContinuousBatch:
    engine = OVGenAI_ContinuousBatch(load_config)
    engine.model = pipeline
    engine.encoder_tokenizer = MagicMock()
    return engine


async def _collect(gen_config: OVGenAI_GenConfig, engine: OVGenAI_ContinuousBatch, events: list, tag: str) -> list:
    items = []
    async for item in engine.generate_type(gen_config):
        items.append(item)
        events.append((tag, item))
    return items


async def _wait_for_handles(pipeline: FakeCBPipeline, count: int) -> None:
    for _ in range(1000):
        if len(pipeline.handles) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {count} handles, have {len(pipeline.handles)}")


@pytest.mark.asyncio
async def test_concurrent_consumers_interleave(load_config: ModelLoadConfig) -> None:
    vocab = {10: "a1", 11: "a2", 20: "b1"}
    gate = threading.Event()
    pipeline = FakeCBPipeline(
        FakeOVTokenizer(vocab),
        scripts={0: [[10], [], [11]], 1: [[], [20]]},
        first_step_gate=gate,
    )
    engine = _make_engine(load_config, pipeline)
    events: list = []

    cfg_a = OVGenAI_GenConfig(stream=True, prompt="prompt-a", request_id="req-a")
    cfg_b = OVGenAI_GenConfig(stream=True, prompt="prompt-b", request_id="req-b")

    task_a = asyncio.create_task(_collect(cfg_a, engine, events, "A"))
    task_b = asyncio.create_task(_collect(cfg_b, engine, events, "B"))
    await _wait_for_handles(pipeline, 2)
    # Deterministic admission order: req-a got internal id 0, req-b id 1.
    assert pipeline.add_calls[0][1] == "prompt-a"
    assert pipeline.add_calls[1][1] == "prompt-b"
    gate.set()

    items_a, items_b = await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=10)

    assert items_a[:2] == ["a1", "a2"]
    assert isinstance(items_a[2], dict)  # trailing metrics
    assert items_b[0] == "b1"
    assert isinstance(items_b[1], dict)

    # No serialization: b1 lands between a1 and a2.
    chunk_events = [(tag, item) for tag, item in events if isinstance(item, str)]
    assert chunk_events[0] == ("A", "a1")
    assert chunk_events.index(("B", "b1")) < chunk_events.index(("A", "a2"))


@pytest.mark.asyncio
async def test_stream_parity_with_chunk_streamer(load_config: ModelLoadConfig) -> None:
    vocab = {i: f"t{i}" for i in range(1, 6)}
    tokenizer = FakeOVTokenizer(vocab)
    id_batches = [[1, 2], [3, 4], [5]]
    pipeline = FakeCBPipeline(tokenizer, scripts={0: id_batches})
    engine = _make_engine(load_config, pipeline)

    gen_config = OVGenAI_GenConfig(
        stream=True, stream_chunk_tokens=2, prompt="p", request_id="req-parity"
    )

    # Reference: drive a ChunkStreamer directly with the same token ids.
    reference = ChunkStreamer(tokenizer, gen_config)
    for batch in id_batches:
        reference.write(batch)
    reference.end()
    await asyncio.sleep(0)  # flush call_soon_threadsafe callbacks
    reference_text = ""
    while True:
        chunk = reference.text_queue.get_nowait()
        if chunk is None:
            break
        reference_text += chunk

    streamed = ""
    async for item in engine.generate_type(gen_config):
        if isinstance(item, str):
            streamed += item

    assert streamed == reference_text
    assert streamed == "t1t2t3t4t5"


@pytest.mark.asyncio
async def test_non_stream_yields_metrics_then_text(load_config: ModelLoadConfig) -> None:
    vocab = {5: "x", 6: "y"}
    pipeline = FakeCBPipeline(FakeOVTokenizer(vocab), scripts={0: [[5, 6]]})
    engine = _make_engine(load_config, pipeline)

    gen_config = OVGenAI_GenConfig(stream=False, prompt="p")
    items = []
    async for item in engine.generate_type(gen_config):
        items.append(item)

    assert len(items) == 2
    metrics, text = items
    assert isinstance(metrics, dict)
    assert metrics["stream"] is False
    assert metrics["new_token"] == 10
    assert text == "xy"


@pytest.mark.asyncio
async def test_cancel_terminates_consumer(load_config: ModelLoadConfig) -> None:
    gate = threading.Event()
    pipeline = FakeCBPipeline(
        FakeOVTokenizer({1: "a"}),
        scripts={0: [[1], [1], [1]]},
        first_step_gate=gate,
    )
    engine = _make_engine(load_config, pipeline)
    events: list = []

    cfg = OVGenAI_GenConfig(stream=True, prompt="p", request_id="req-cancel")
    task = asyncio.create_task(_collect(cfg, engine, events, "A"))
    await _wait_for_handles(pipeline, 1)

    assert await engine.cancel("req-cancel") is True
    assert pipeline.handles[0].cancelled is True
    gate.set()

    items = await asyncio.wait_for(task, timeout=10)
    # Consumer terminated (no hang); a metrics dict closes the stream.
    assert isinstance(items[-1], dict)
    assert await engine.cancel("req-cancel") is False  # already gone


@pytest.mark.asyncio
async def test_ignored_request_errors_other_continues(load_config: ModelLoadConfig) -> None:
    gate = threading.Event()
    vocab = {7: "z1", 8: "z2"}
    pipeline = FakeCBPipeline(
        FakeOVTokenizer(vocab),
        scripts={0: ["IGNORED"], 1: [[7], [8]]},
        first_step_gate=gate,
    )
    engine = _make_engine(load_config, pipeline)
    events: list = []

    cfg_a = OVGenAI_GenConfig(stream=True, prompt="prompt-a", request_id="req-ignored")
    cfg_b = OVGenAI_GenConfig(stream=True, prompt="prompt-b", request_id="req-ok")

    async def _collect_raising():
        return await _collect(cfg_a, engine, events, "A")

    task_a = asyncio.create_task(_collect_raising())
    task_b = asyncio.create_task(_collect(cfg_b, engine, events, "B"))
    await _wait_for_handles(pipeline, 2)
    gate.set()

    with pytest.raises(RuntimeError, match="ignored by the continuous batching scheduler"):
        await asyncio.wait_for(task_a, timeout=10)

    items_b = await asyncio.wait_for(task_b, timeout=10)
    assert items_b[:2] == ["z1", "z2"]
    assert isinstance(items_b[2], dict)


@pytest.mark.asyncio
async def test_perf_metrics_raise_on_cancel_still_terminates(load_config: ModelLoadConfig) -> None:
    gate = threading.Event()
    pipeline = FakeCBPipeline(
        FakeOVTokenizer({1: "a"}),
        scripts={0: [[1], [1]]},
        first_step_gate=gate,
        perf_raises_ids={0},
    )
    engine = _make_engine(load_config, pipeline)
    events: list = []

    cfg = OVGenAI_GenConfig(stream=True, prompt="p", request_id="req-cancel-perf")
    task = asyncio.create_task(_collect(cfg, engine, events, "A"))
    await _wait_for_handles(pipeline, 1)

    await engine.cancel("req-cancel-perf")
    gate.set()

    items = await asyncio.wait_for(task, timeout=10)
    # Fallback metrics (get_perf_metrics threw on the CANCEL handle).
    assert items[-1] == {"stream": True}


@pytest.mark.asyncio
async def test_unload_with_in_flight_requests(load_config: ModelLoadConfig) -> None:
    gate = threading.Event()
    pipeline = FakeCBPipeline(
        FakeOVTokenizer({1: "a"}),
        scripts={0: [[1], [1], [1]]},
        first_step_gate=gate,
    )
    engine = _make_engine(load_config, pipeline)
    events: list = []

    cfg = OVGenAI_GenConfig(stream=True, prompt="p", request_id="req-inflight")
    task = asyncio.create_task(_collect(cfg, engine, events, "A"))
    await _wait_for_handles(pipeline, 1)

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    unload_task = asyncio.create_task(engine.unload_model(registry, "test-cb-model"))
    await asyncio.sleep(0.05)  # let unload reach the blocked step task
    gate.set()

    removed = await asyncio.wait_for(unload_task, timeout=10)
    assert removed is True
    registry.register_unload.assert_called_once_with("test-cb-model")

    # The in-flight consumer terminates instead of hanging.
    await asyncio.wait_for(task, timeout=10)

    assert engine.model is None
    assert engine.encoder_tokenizer is None
    assert engine._step_task is None
    assert engine._executor is None


def test_load_model_constructs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARC_WARMUP", "0")
    pipeline_instance = MagicMock()
    pipeline_factory = MagicMock(return_value=pipeline_instance)
    monkeypatch.setattr(cb_module, "ContinuousBatchingPipeline", pipeline_factory)
    tokenizer_instance = MagicMock()
    monkeypatch.setattr(
        cb_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer_instance),
    )

    loader = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI_CB,
        device="CPU",
        runtime_config={"hint": "value"},
        cache_dir="/tmp/ov_cache",
    )

    engine = OVGenAI_ContinuousBatch(loader)
    engine.load_model(loader)

    pipeline_factory.assert_called_once()
    args, kwargs = pipeline_factory.call_args
    assert args == (loader.model_path,)
    assert kwargs["device"] == loader.device
    assert kwargs["properties"]["hint"] == "value"
    assert kwargs["properties"]["CACHE_DIR"] == "/tmp/ov_cache"
    # PA-backend default: prefix caching on.
    assert kwargs["scheduler_config"].enable_prefix_caching is True
    cb_module.AutoTokenizer.from_pretrained.assert_called_once_with(loader.model_path)
    assert engine.model is pipeline_instance
    assert engine.encoder_tokenizer is tokenizer_instance


def test_load_model_forwards_draft_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARC_WARMUP", "0")
    pipeline_factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(cb_module, "ContinuousBatchingPipeline", pipeline_factory)
    monkeypatch.setattr(
        cb_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    draft = object()
    draft_factory = MagicMock(return_value=draft)
    monkeypatch.setattr(utils_module.openvino_genai, "draft_model", draft_factory)

    loader = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI_CB,
        device="CPU",
        runtime_config={},
        cache_dir="/tmp/ov_cache",
        draft_model_path="/models/draft",
        draft_device="CPU",
    )

    OVGenAI_ContinuousBatch(loader).load_model(loader)

    draft_factory.assert_called_once_with("/models/draft", "CPU", CACHE_DIR="/tmp/ov_cache")
    _, kwargs = pipeline_factory.call_args
    assert kwargs["properties"]["draft_model"] is draft


def test_load_model_vlm_resolves_vision_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARC_WARMUP", "0")
    monkeypatch.setattr(cb_module, "ContinuousBatchingPipeline", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        cb_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    vision_mock = MagicMock(return_value="<|image_pad|>")
    monkeypatch.setattr(cb_module, "resolve_vlm_vision_token", vision_mock)

    loader = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-vlm",
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI_CB,
        device="CPU",
        runtime_config={},
    )

    engine = OVGenAI_ContinuousBatch(loader)
    engine.load_model(loader)

    vision_mock.assert_called_once_with(loader.model_path)
    assert engine.vision_token == "<|image_pad|>"


@pytest.mark.asyncio
async def test_vlm_admit_uses_prompt_overload(vlm_load_config: ModelLoadConfig) -> None:
    pipeline = FakeCBPipeline(FakeOVTokenizer({1: "a"}), scripts={0: [[1]]})
    engine = _make_engine(vlm_load_config, pipeline)
    engine.vision_token = "<|image_pad|>"

    gen_config = OVGenAI_GenConfig(stream=False, prompt="describe this")
    items = []
    async for item in engine.generate_type(gen_config):
        items.append(item)

    # (internal_id, prompt, generation_config): no images -> str-prompt overload.
    call = pipeline.add_calls[0]
    assert call[0] == 0
    assert call[1] == "describe this"
    assert len(items) == 2


@pytest.mark.asyncio
async def test_vlm_admit_strips_stray_vision_token(vlm_load_config: ModelLoadConfig) -> None:
    pipeline = FakeCBPipeline(FakeOVTokenizer({1: "a"}), scripts={0: [[1]]})
    engine = _make_engine(vlm_load_config, pipeline)
    engine.vision_token = "<|image_pad|>"

    gen_config = OVGenAI_GenConfig(stream=False, prompt="what does <|image_pad|> mean?")
    async for _ in engine.generate_type(gen_config):
        pass

    call = pipeline.add_calls[0]
    assert "<|image_pad|>" not in call[1]


def test_create_generation_config_zero_temperature_disables_sampling(
    monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig
) -> None:
    class DummyGenerationConfig:
        pass

    monkeypatch.setattr(cb_module, "GenerationConfig", DummyGenerationConfig)
    engine = OVGenAI_ContinuousBatch(load_config)
    engine.model = None

    config = engine.create_generation_config(OVGenAI_GenConfig(temperature=0.0))

    assert config.do_sample is False
    assert config.temperature == 0.0
    assert config.apply_chat_template is False


def test_registry_maps_cb_engine() -> None:
    assert (
        MODEL_CLASS_REGISTRY[(EngineType.OV_GENAI_CB, ModelType.LLM)]
        == "src.engine.ov_genai.continuous_batch_llm.OVGenAI_ContinuousBatch"
    )
    assert (
        MODEL_CLASS_REGISTRY[(EngineType.OV_GENAI_CB, ModelType.VLM)]
        == "src.engine.ov_genai.continuous_batch_llm.OVGenAI_ContinuousBatch"
    )
