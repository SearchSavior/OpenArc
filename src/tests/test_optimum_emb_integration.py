import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import PreTrainedTokenizerConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/Pytorch/Qwen/Qwen3-Embed-0.6B-INT8-ASYM-ov")
BGE_M3_PATH = Path("/data/openvino-models/embeddings/bge-m3")
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


def test_bge_m3_cls_pool_cpu_integration() -> None:
    """Load the converted bge-m3 OV IR and verify CLS pooling is auto-selected.

    The sentence-transformers metadata shipped with bge-m3 declares CLS pooling,
    so loading without a runtime_config override should pick it up automatically
    and emit a unit-normed 1024-dim vector.
    """
    _ensure_unit_tests_pass()
    if not BGE_M3_PATH.exists():
        pytest.skip(f"bge-m3 OV IR not found at {BGE_M3_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(BGE_M3_PATH),
        model_name="integration-bge-m3",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    emb = Optimum_EMB(load_config)
    emb.load_model(load_config)
    try:
        assert emb.pool_mode == "cls", f"expected cls auto-detect, got {emb.pool_mode!r}"

        tok_config = PreTrainedTokenizerConfig(
            text=["What is the capital of France?"],
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        async def _run():
            vectors = []
            async for item in emb.generate_embeddings(tok_config):
                vectors.append(item)
            return vectors

        outputs = asyncio.run(_run())
        assert len(outputs) == 1
        vec = outputs[0]
        assert isinstance(vec, list) and len(vec) == 1
        assert len(vec[0]) == 1024
        import math
        norm = math.sqrt(sum(x * x for x in vec[0]))
        assert abs(norm - 1.0) < 1e-3, f"expected unit-normed, got ||v||={norm}"
    finally:
        asyncio.run(emb.unload_model(_DummyRegistry(), load_config.model_name))


def test_bge_m3_runtime_config_override_forces_last_pool() -> None:
    """runtime_config `pool_mode` override beats the shipped sentence-transformers metadata."""
    _ensure_unit_tests_pass()
    if not BGE_M3_PATH.exists():
        pytest.skip(f"bge-m3 OV IR not found at {BGE_M3_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(BGE_M3_PATH),
        model_name="integration-bge-m3-override",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={"pool_mode": "last"},
    )

    emb = Optimum_EMB(load_config)
    emb.load_model(load_config)
    try:
        assert emb.pool_mode == "last"
    finally:
        asyncio.run(emb.unload_model(_DummyRegistry(), load_config.model_name))
