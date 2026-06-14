import base64
import io
import wave

import numpy as np
import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.ov_genai.whisper import OVGenAI_Whisper
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_WhisperGenConfig


MODEL_PATH = model_path("whisper-tiny-int8-ov")


def _generate_sine_wave_base64(duration_s: float = 0.5, sr: int = 16000) -> str:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = (samples * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True

@pytest.mark.asyncio
async def test_whisper_transcribe_cpu_integration() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-whisper",
        model_type=ModelType.WHISPER,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    whisper = OVGenAI_Whisper(load_config)
    whisper.load_model(load_config)

    try:
        audio_base64 = _generate_sine_wave_base64()
        gen_config = OVGenAI_WhisperGenConfig(audio_base64=audio_base64)


        outputs = []
        async for item in whisper.transcribe(gen_config):
            outputs.append(item)

        assert len(outputs) == 2
        metrics, text = outputs
        assert isinstance(metrics, dict)
        assert isinstance(text, str)
        assert text.strip(), "Expected non-empty transcription"

    finally:
        await whisper.unload_model(_DummyRegistry(), load_config.model_name)

