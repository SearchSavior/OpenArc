import asyncio

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import PreTrainedTokenizerConfig

MODEL_PATH = model_path("Qwen3-Embedding-0.6B-int8_asym-ov")

class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_optimum_emb_generate_embeddings_cpu_integration() -> None:
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

