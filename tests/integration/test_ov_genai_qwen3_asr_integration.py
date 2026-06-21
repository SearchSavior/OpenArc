import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import]

from src.server.models.openvino import OV_Qwen3ASRGenConfig
from src.engine.openvino.qwen3_asr.qwen3_asr import OVQwen3ASR
from test_model_path import model_path
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType



MODEL_PATH = model_path("Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO")
AUDIO_PATH = Path(__file__).parents[1] / "litany_against_fear_dune.wav"

@dataclass(frozen=True)
class Qwen3ASRResult:
    text: str
    metrics: dict[str, Any]
    segments: list[Any]

def _read_audio_base64(audio_path: Path) -> str:
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")

class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


@pytest.fixture(scope="module")
def qwen3asr_result() -> Qwen3ASRResult:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    if not AUDIO_PATH.exists():
        pytest.skip(f"Audio fixture not found: {AUDIO_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-qwen3asr",
        model_type=ModelType.QWEN3_ASR,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    asr = OVQwen3ASR(load_config)
    asr.load_model(load_config)

    try:
        audio_base64 = _read_audio_base64(AUDIO_PATH)
        gen_config = OV_Qwen3ASRGenConfig(audio_base64=audio_base64)

        text, metrics, segments = asyncio.run(asr.transcribe(gen_config))

        return Qwen3ASRResult(
            text=text,
            metrics=metrics,
            segments=segments,
        )

    finally:
        asyncio.run(asr.unload_model(_DummyRegistry(), load_config.model_name))


def test_qwen3asr_transcribe_returns_metrics(qwen3asr_result: Qwen3ASRResult) -> None:
    assert isinstance(qwen3asr_result.metrics, dict)


def test_qwen3asr_transcribe_returns_text(qwen3asr_result: Qwen3ASRResult) -> None:
    assert isinstance(qwen3asr_result.text, str)


def test_qwen3asr_transcribe_returns_segments(qwen3asr_result: Qwen3ASRResult) -> None:
    assert isinstance(qwen3asr_result.segments, list)


def test_qwen3asr_reports_end_to_end_time(qwen3asr_result: Qwen3ASRResult) -> None:
    assert "end_to_end_sec" in qwen3asr_result.metrics


def test_qwen3asr_reports_audio_duration_correctly(qwen3asr_result: Qwen3ASRResult) -> None:
    assert "audio_duration_sec" in qwen3asr_result.metrics
    assert qwen3asr_result.metrics["audio_duration_sec"] == pytest.approx(21, abs=0.5)


def test_qwen3asr_transcribe_includes_total_obliteration(qwen3asr_result: Qwen3ASRResult) -> None:
    assert "total obliteration" in qwen3asr_result.text.lower()


def test_qwen3asr_transcribe_includes_i_will_face_my_fear(qwen3asr_result: Qwen3ASRResult) -> None:
    assert "i will face my fear" in qwen3asr_result.text.lower()