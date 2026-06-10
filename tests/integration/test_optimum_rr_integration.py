import asyncio

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.optimum.optimum_rr import Optimum_RR
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import RerankerConfig

MODEL_PATH = model_path("Qwen3-Reranker-0.6B-fp16-ov")

class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True


def test_optimum_rr_generate_rerankings_cpu_integration() -> None:
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

