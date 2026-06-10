import base64
from pathlib import Path

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = model_path("Qwen2.5-VL-3B-Instruct-int4_sym-ov")

TEST_IMAGE_PATH = Path(__file__).parents[1] / "dedication.png"
def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _make_messages(base64_image: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image succinctly."},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ],
        }
    ]


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:  # noqa: D401 - simple stub
        return True

@pytest.mark.asyncio
async def test_vlm_generate_text_cpu_integration() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    load_config = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="integration-vlm",
        model_type=ModelType.VLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
        vlm_type="qwen25vl",
    )

    vlm = OVGenAI_VLM(load_config)
    vlm.load_model(load_config)

    try:
        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        base64_image = _image_data_url(TEST_IMAGE_PATH)

        gen_config = OVGenAI_GenConfig(
            messages=_make_messages(base64_image),
            max_tokens=16,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            stream=False,
        )

        outputs = []
        async for item in vlm.generate_text(gen_config):
            outputs.append(item)

        assert len(outputs) == 2
        metrics, text = outputs
        assert isinstance(metrics, dict)
        assert metrics["stream"] is False
        assert isinstance(text, str)
        assert text.strip(), "Expected non-empty model response"

    finally:
        await vlm.unload_model(_DummyRegistry(), load_config.model_name)
