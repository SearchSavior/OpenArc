import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.ov_genai.llm import OVGenAI_LLM
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = model_path("Qwen3-0.6B-int8-ov")

class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True

@pytest.mark.asyncio
async def test_llm_generate_text_cpu_integration() -> None:
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

        outputs = []
        async for item in llm.generate_text(gen_config):
            outputs.append(item)

        assert len(outputs) == 2
        metrics, text = outputs
        assert isinstance(metrics, dict)
        assert metrics["stream"] is False
        assert isinstance(text, str)
        assert text.strip(), "Expected non-empty model response"

    finally:
        await llm.unload_model(_DummyRegistry(), load_config.model_name)

