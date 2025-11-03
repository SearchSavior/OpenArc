import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.optimum.optimum_rr import Optimum_RR
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import RerankerConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-Reranker-0.6B-fp16-ov")
UNIT_TEST_PATH = Path(__file__).with_name("test_optimum_rr_unit.py")

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
            "Skipping reranker integration test because unit tests failed:\n" + _UNIT_TEST_OUTPUT
        )


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_optimum_rr_generate_rerankings_cpu_integration() -> None:
    _ensure_unit_tests_pass()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-rerank",
        model_type=ModelType.RERANK,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    rr = Optimum_RR(load_config)
    rr.load_model(load_config)

    try:
        config = RerankerConfig(
            query="Which document mentions Paris?",
            documents=[
                "Paris is the capital of France and a popular tourist destination.",
                "Berlin is known for its art scene and modern landmarks.",
            ],
            prefix="",
            suffix="",
            instruction="Select the document relevant to the query.",
            max_length=256,
        )

        async def _run():
            results = []
            async for item in rr.generate_rerankings(config):
                results.append(item)
            return results

        outputs = asyncio.run(_run())

        assert outputs, "Expected reranker output"
        ranked = outputs[0]
        assert len(ranked) == len(config.documents)
        assert ranked[0]["score"] >= ranked[1]["score"]
        assert "Paris" in ranked[0]["doc"]

    finally:
        asyncio.run(rr.unload_model(_DummyRegistry(), load_config.model_name))

