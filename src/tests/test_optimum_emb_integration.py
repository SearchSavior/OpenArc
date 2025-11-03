import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import PreTrainedTokenizerConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/Pytorch/Qwen/Qwen3-Embed-0.6B-INT8-ASYM-ov")
UNIT_TEST_PATH = Path(__file__).with_name("test_optimum_emb_unit.py")

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
            "Skipping embedding integration test because unit tests failed:\n" + _UNIT_TEST_OUTPUT
        )


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_optimum_emb_generate_embeddings_cpu_integration() -> None:
    _ensure_unit_tests_pass()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-emb",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    emb = Optimum_EMB(load_config)
    emb.load_model(load_config)

    try:
        tok_config = PreTrainedTokenizerConfig(
            text=["OpenVINO embedding integration test."],
            padding="longest",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        async def _run():
            vectors = []
            async for item in emb.generate_embeddings(tok_config):
                vectors.append(item)
            return vectors

        outputs = asyncio.run(_run())

        assert len(outputs) == 1
        embedding = outputs[0]
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    finally:
        asyncio.run(emb.unload_model(_DummyRegistry(), load_config.model_name))

