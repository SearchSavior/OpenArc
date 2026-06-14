import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest  # type: ignore[import]

import src.engine.ov_genai.whisper as whisper_module
from src.engine.ov_genai.whisper import OVGenAI_Whisper
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_WhisperGenConfig


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/distil-whisper-large-v3-int8-ov"


class DummyMean:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class DummyPerfMetrics:
    def get_num_generated_tokens(self) -> int:
        return 42

    def get_throughput(self) -> DummyMean:
        return DummyMean(3.1415)

    def get_ttft(self) -> DummyMean:
        return DummyMean(250.0)

    def get_load_time(self) -> int:
        return 5000

    def get_generate_duration(self) -> DummyMean:
        return DummyMean(2000.0)

    def get_features_extraction_duration(self) -> DummyMean:
        return DummyMean(12.5)


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-whisper",
        model_type=ModelType.WHISPER,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"config": "value"},
    )


def _sample_audio_base64() -> str:
    return base64.b64encode(b"audio-bytes").decode("ascii")


def test_prepare_audio_calls_librosa(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    whisper = OVGenAI_Whisper(load_config)
    audio_array = np.array([0.1, -0.2], dtype=np.float32)

    load_mock = MagicMock(return_value=(audio_array, 16000))
    monkeypatch.setattr(whisper_module.librosa, "load", load_mock)

    audio_list = whisper.prepare_audio(OVGenAI_WhisperGenConfig(audio_base64=_sample_audio_base64()))

    load_mock.assert_called_once()
    _, kwargs = load_mock.call_args
    assert kwargs["sr"] == 16000
    assert kwargs["mono"] is True
    assert audio_list == audio_array.tolist()


def test_collect_metrics_formats_values(load_config: ModelLoadConfig) -> None:
    whisper = OVGenAI_Whisper(load_config)
    metrics = whisper.collect_metrics(DummyPerfMetrics())

    assert metrics["num_generated_tokens"] == 42
    assert metrics["throughput_tokens_per_sec"] == round(3.1415, 4)
    assert metrics["ttft_s"] == round(0.25, 4)
    assert metrics["load_time_s"] == round(5.0, 4)
    assert metrics["generate_duration_s"] == round(2.0, 4)
    assert metrics["features_extraction_duration_ms"] == round(12.5, 4)


def test_transcribe_yields_metrics_and_text(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    whisper = OVGenAI_Whisper(load_config)

    async def immediate_to_thread(func, *args, **kwargs):  # type: ignore[override]
        return func(*args, **kwargs)

    monkeypatch.setattr(whisper_module.asyncio, "to_thread", immediate_to_thread)

    prepare_mock = MagicMock(return_value=[0.1, 0.2])
    whisper.prepare_audio = prepare_mock

    result = MagicMock()
    result.texts = ["hello", "world"]
    result.perf_metrics = DummyPerfMetrics()

    whisper.whisper_model = MagicMock()
    whisper.whisper_model.generate.return_value = result

    gen_config = OVGenAI_WhisperGenConfig(audio_base64=_sample_audio_base64())

    async def _run_test():
        outputs = []
        async for item in whisper.transcribe(gen_config):
            outputs.append(item)
        return outputs

    outputs = asyncio.run(_run_test())

    assert len(outputs) == 2
    metrics, text = outputs
    assert isinstance(metrics, dict)
    assert text == "hello world"
    whisper.whisper_model.generate.assert_called_once_with([0.1, 0.2])
    prepare_mock.assert_called_once_with(gen_config)


def test_load_model_initializes_pipeline(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    whisper = OVGenAI_Whisper(load_config)
    pipeline_instance = MagicMock()
    pipeline_factory = MagicMock(return_value=pipeline_instance)
    monkeypatch.setattr(whisper_module, "WhisperPipeline", pipeline_factory)

    whisper.load_model(load_config)

    pipeline_factory.assert_called_once_with(
        load_config.model_path,
        load_config.device,
        **load_config.runtime_config,
    )
    assert whisper.whisper_model is pipeline_instance


def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    whisper = OVGenAI_Whisper(load_config)
    whisper.whisper_model = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(whisper_module.gc, "collect", gc_mock)

    result = asyncio.run(whisper.unload_model(registry, "model-name"))

    assert result is True
    assert whisper.whisper_model is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

