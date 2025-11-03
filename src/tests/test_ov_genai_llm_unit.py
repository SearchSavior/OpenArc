from unittest.mock import AsyncMock, MagicMock

import pytest  # type: ignore[import]

import src.engine.ov_genai.llm as llm_module
from src.engine.ov_genai.llm import OVGenAI_LLM
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-Reranker-0.6B-fp16-ov"


class DummyMeanValue:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class DummyPerfMetrics:
    def __init__(self) -> None:
        self._input_tokens = 25
        self._generated_tokens = 10

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
        return self._input_tokens

    def get_num_generated_tokens(self) -> int:
        return self._generated_tokens


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"config": "value"},
    )


def test_prepare_inputs_passes_tools(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "tool", "description": "desc"}]

    apply_mock = MagicMock(return_value="np_payload")
    llm.encoder_tokenizer = MagicMock()
    llm.encoder_tokenizer.apply_chat_template = apply_mock

    class DummyTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(llm_module.ov, "Tensor", DummyTensor)

    result = llm.prepare_inputs(messages, tools)

    assert isinstance(result, DummyTensor)
    apply_mock.assert_called_once_with(
        messages,
        tools=tools,
        add_generation_prompt=True,
        skip_special_tokens=True,
        return_tensors="np",
    )


def test_prepare_inputs_without_tools_uses_none(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    messages = [{"role": "user", "content": "hi"}]

    apply_mock = MagicMock(return_value="np_payload")
    llm.encoder_tokenizer = MagicMock()
    llm.encoder_tokenizer.apply_chat_template = apply_mock

    class DummyTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(llm_module.ov, "Tensor", DummyTensor)

    llm.prepare_inputs(messages)

    apply_mock.assert_called_once_with(
        messages,
        tools=None,
        add_generation_prompt=True,
        skip_special_tokens=True,
        return_tensors="np",
    )


@pytest.mark.parametrize(
    ("stream", "target"),
    (
        (True, "generate_stream"),
        (False, "generate_text"),
    ),
)
def test_generate_type_respects_stream_flag(load_config: ModelLoadConfig, stream: bool, target: str) -> None:
    llm = OVGenAI_LLM(load_config)
    gen_config = OVGenAI_GenConfig(stream=stream)

    expected = object()
    setattr(llm, target, MagicMock(return_value=expected))

    result = llm.generate_type(gen_config)

    assert result is expected
    getattr(llm, target).assert_called_once_with(gen_config)


def test_collect_metrics_prefill_throughput(load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    gen_config = OVGenAI_GenConfig(stream=True, stream_chunk_tokens=2)

    metrics = llm.collect_metrics(gen_config, DummyPerfMetrics())

    assert metrics["prefill_throughput (tokens/s)"] == 100.0
    assert metrics["stream"] is True
    assert metrics["stream_chunk_tokens"] == 2


def test_load_model_sets_pipeline_and_tokenizer(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    pipeline_instance = MagicMock()
    pipeline_factory = MagicMock(return_value=pipeline_instance)
    monkeypatch.setattr(llm_module, "LLMPipeline", pipeline_factory)

    tokenizer_instance = MagicMock()
    monkeypatch.setattr(
        llm_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer_instance),
    )

    loader = ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"hint": "value"},
    )

    llm.load_model(loader)

    pipeline_factory.assert_called_once_with(
        loader.model_path,
        loader.device,
        **loader.runtime_config,
    )
    llm_module.AutoTokenizer.from_pretrained.assert_called_once_with(loader.model_path)
    assert llm.model is pipeline_instance
    assert llm.encoder_tokenizer is tokenizer_instance


@pytest.mark.asyncio
async def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    llm.model = object()
    llm.encoder_tokenizer = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(llm_module.gc, "collect", gc_mock)

    result = await llm.unload_model(registry, "model-name")

    assert result is True
    assert llm.model is None
    assert llm.encoder_tokenizer is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

