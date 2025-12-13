import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]
import torch

from src.engine.openvino.kokoro import OV_Kokoro
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.openvino import KokoroLanguage, KokoroVoice, OV_KokoroGenConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO")
UNIT_TEST_PATH = Path(__file__).with_name("test_ov_genai_kokoro_unit.py")

_UNIT_TESTS_PASSED: bool | None = None
_UNIT_TEST_OUTPUT: str = ""


def _ensure_unit_tests_pass() -> None:
    global _UNIT_TESTS_PASSED, _UNIT_TEST_OUTPUT

    if _UNIT_TESTS_PASSED is None:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(UNIT_TEST_PATH), "-q"],
            capture_output=True,
            text=True,
        )
        _UNIT_TESTS_PASSED = result.returncode == 0
        _UNIT_TEST_OUTPUT = (result.stdout or "") + (result.stderr or "")

    if not _UNIT_TESTS_PASSED:
        pytest.skip(
            "Skipping Kokoro integration test because unit tests failed:\n" + _UNIT_TEST_OUTPUT
        )


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_kokoro_chunk_forward_pass_cpu_integration() -> None:
    _ensure_unit_tests_pass()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-kokoro",
        model_type=ModelType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",
        runtime_config={},
    )

    kokoro = OV_Kokoro(load_config)
    kokoro.load_model(load_config)

    try:
        gen_config = OV_KokoroGenConfig(
            kokoro_message="Hello world from Kokoro.",
            voice=KokoroVoice.AF_SARAH,
            lang_code=KokoroLanguage.AMERICAN_ENGLISH,
            speed=1.0,
            character_count_chunk=120,
            response_format="wav",
        )

        async def _run_test():
            chunks = []
            async for chunk in kokoro.chunk_forward_pass(gen_config):
                chunks.append(chunk)
                break
            return chunks

        chunks = asyncio.run(_run_test())

        assert chunks, "Expected at least one audio chunk"
        first_chunk = chunks[0]
        assert isinstance(first_chunk.chunk_text, str)
        assert first_chunk.chunk_text
        assert isinstance(first_chunk.audio, torch.Tensor)
        assert first_chunk.audio.numel() > 0

    finally:
        asyncio.run(kokoro.unload_model(_DummyRegistry(), load_config.model_name))

