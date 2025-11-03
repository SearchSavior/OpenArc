from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest  # type: ignore[import]

import src.engine.ov_genai.vlm as vlm_module
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-3B-Instruct-int4_sym-ov"


class DummyMeanValue:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class DummyPerfMetrics:
    def __init__(self) -> None:
        self._input_tokens = 12
        self._generated_tokens = 4

    def get_load_time(self) -> int:
        return 8000

    def get_ttft(self) -> DummyMeanValue:
        return DummyMeanValue(400.0)

    def get_tpot(self) -> DummyMeanValue:
        return DummyMeanValue(6.0)

    def get_throughput(self) -> DummyMeanValue:
        return DummyMeanValue(9.87654)

    def get_generate_duration(self) -> DummyMeanValue:
        return DummyMeanValue(2000.0)

    def get_num_input_tokens(self) -> int:
        return self._input_tokens

    def get_num_generated_tokens(self) -> int:
        return self._generated_tokens


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-vlm",
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"config": "value"},
        vlm_type="internvl2",
    )


def test_vision_token_placeholder_replaced(load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    vlm.vision_token = "<|image_{i}|>"

    assert vlm._vision_token_for_index(3) == "<|image_3|>"


def test_vision_token_without_placeholder(load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    vlm.vision_token = "<image>"

    assert vlm._vision_token_for_index(1) == "<image>"


def test_prepare_inputs_extracts_image_and_text(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    vlm.vision_token = "<|image_{i}|>"
    tokenizer_mock = MagicMock()
    tokenizer_mock.apply_chat_template.return_value = "[templated]"
    vlm.tokenizer = tokenizer_mock

    class DummyTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(vlm_module.ov, "Tensor", DummyTensor)

    class FakeImage:
        def convert(self, _mode):
            return self

        def __array__(self, dtype=None):
            return np.zeros((1, 1, 3), dtype=dtype or np.uint8)

    def fake_open(_bytes):
        return FakeImage()

    monkeypatch.setattr(vlm_module.Image, "open", fake_open)

    base64_pixel = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_pixel}},
                {"type": "text", "text": "Describe"},
            ],
        }
    ]
    tools = [{"name": "tool"}]

    prompt, ov_images = vlm.prepare_inputs(messages, tools)

    assert prompt == "[templated]"
    assert len(ov_images) == 1
    assert isinstance(ov_images[0], DummyTensor)

    args, kwargs = tokenizer_mock.apply_chat_template.call_args
    assert "<|image_0|>" in args[0][0]["content"]
    assert kwargs["tools"] is tools


@pytest.mark.parametrize(
    ("stream", "target"),
    (
        (True, "generate_stream"),
        (False, "generate_text"),
    ),
)
def test_generate_type_respects_stream_flag(load_config: ModelLoadConfig, stream: bool, target: str) -> None:
    vlm = OVGenAI_VLM(load_config)
    config = OVGenAI_GenConfig(stream=stream)

    expected = object()
    setattr(vlm, target, MagicMock(return_value=expected))

    result = vlm.generate_type(config)

    assert result is expected
    getattr(vlm, target).assert_called_once_with(config)


def test_collect_metrics_prefill_throughput(load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    config = OVGenAI_GenConfig(stream=True, stream_chunk_tokens=2)

    metrics = vlm.collect_metrics(config, DummyPerfMetrics())

    assert metrics["prefill_throughput (tokens/s)"] == 30.0
    assert metrics["stream"] is True
    assert metrics["stream_chunk_tokens"] == 2


def test_load_model_sets_pipeline_and_vision_token(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    pipeline_instance = MagicMock()
    pipeline_factory = MagicMock(return_value=pipeline_instance)
    monkeypatch.setattr(vlm_module, "VLMPipeline", pipeline_factory)

    tokenizer_instance = MagicMock()
    monkeypatch.setattr(
        vlm_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer_instance),
    )

    vlm.load_model(load_config)

    pipeline_factory.assert_called_once_with(
        load_config.model_path,
        load_config.device,
        **load_config.runtime_config,
    )
    vlm_module.AutoTokenizer.from_pretrained.assert_called_once_with(load_config.model_path)
    assert vlm.model_path is pipeline_instance
    assert vlm.tokenizer is tokenizer_instance
    assert vlm.vision_token == vlm_module.VLM_VISION_TOKENS[load_config.vlm_type]


@pytest.mark.asyncio
async def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    vlm = OVGenAI_VLM(load_config)
    vlm.model_path = object()
    vlm.tokenizer = object()
    vlm.vision_token = "token"

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(vlm_module.gc, "collect", gc_mock)

    result = await vlm.unload_model(registry, "model-name")

    assert result is True
    assert vlm.model_path is None
    assert vlm.tokenizer is None
    assert vlm.vision_token is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

