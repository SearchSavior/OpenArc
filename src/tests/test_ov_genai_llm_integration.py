import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.ov_genai.llm import OVGenAI_LLM
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-Reranker-0.6B-fp16-ov")
UNIT_TEST_PATH = Path(__file__).with_name("test_ov_genai_llm_unit.py")

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
            "Skipping LLM integration test because unit tests failed:\n" + _UNIT_TEST_OUTPUT
        )


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_llm_generate_text_cpu_integration() -> None:
    _ensure_unit_tests_pass()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-llm",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )

    llm = OVGenAI_LLM(load_config)
    llm.load_model(load_config)

    try:
        gen_config = OVGenAI_GenConfig(
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello using fewer than five words."},
            ],
            max_tokens=16,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            stream=False,
        )

        async def _run_test():
            outputs = []
            async for item in llm.generate_text(gen_config):
                outputs.append(item)
            return outputs

        outputs = asyncio.run(_run_test())

        assert len(outputs) == 2
        metrics, text = outputs
        assert isinstance(metrics, dict)
        assert metrics["stream"] is False
        assert isinstance(text, str)
        assert text.strip(), "Expected non-empty model response"

    finally:
        asyncio.run(llm.unload_model(_DummyRegistry(), load_config.model_name))

