import asyncio
import math

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType
from src.server.schemas.modeling.contract_optimum_emb import PreTrainedTokenizerConfig

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
        # A batch of different lengths, padded like the /v1/embeddings defaults. Covers both IR
        # layouts: decoder exports that take position_ids (optimum-intel < 1.26) and newer ones.
        texts = ["OpenVINO embedding integration test.", "short", "a longer second text that needs padding"]
        tok_config = PreTrainedTokenizerConfig(
            text=texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        async def _run():
            vectors = []
            async for item in emb.generate_embeddings(tok_config):
                vectors.append(item)
            return vectors

        outputs = asyncio.run(_run())

        assert outputs[0]["input_token"] > 0
        vectors = outputs[-1]
        assert len(vectors) == len(texts)
        for vector in vectors:
            assert len(vector) > 0
            # issue #106: without position_ids, CPU gave NaN or a missing-input error
            assert all(math.isfinite(value) for value in vector)
            assert abs(math.sqrt(sum(value * value for value in vector)) - 1.0) < 1e-3

        # padding must not change a text's vector
        alone = PreTrainedTokenizerConfig(text=[texts[1]], padding=True, truncation=True, max_length=8192)

        async def _run_alone():
            return [item async for item in emb.generate_embeddings(alone)]

        single = asyncio.run(_run_alone())[-1][0]
        dot = sum(a * b for a, b in zip(single, vectors[1]))
        assert dot > 0.999

    finally:
        asyncio.run(emb.unload_model(_DummyRegistry(), load_config.model_name))

