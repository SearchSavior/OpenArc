import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = model_path("Qwen2.5-VL-3B-Instruct-int4_sym-ov")
TEST_IMAGE_PATH = Path(__file__).parents[1] / "dedication.png"

@dataclass(frozen=True)
class VLMResult:
    text: str
    metrics: dict[str, Any]

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


async def _generate_vlm_result(vlm: OVGenAI_VLM, gen_config: OVGenAI_GenConfig) -> VLMResult:
    outputs = []
    async for item in vlm.generate_text(gen_config):
        outputs.append(item)
    assert len(outputs) == 2
    metrics, text = outputs
    return VLMResult(
        text=text,
        metrics=metrics,
    )


@pytest.fixture(scope="module")
def vlm_generate_text_cpu_integration() -> VLMResult:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path not found: {MODEL_PATH}")

    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image fixture not found: {TEST_IMAGE_PATH}")

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
        base64_image = _image_data_url(TEST_IMAGE_PATH)

        gen_config = OVGenAI_GenConfig(
            messages=_make_messages(base64_image),
            max_tokens=16,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            stream=False,
        )

        return asyncio.run(_generate_vlm_result(vlm, gen_config))

    finally:
        asyncio.run(vlm.unload_model(_DummyRegistry(), load_config.model_name))


def test_generated_result_has_text_string(vlm_generate_text_cpu_integration: VLMResult) -> None:
    assert isinstance(vlm_generate_text_cpu_integration.text, str)

def test_generated_result_has_metrics_dict(vlm_generate_text_cpu_integration: VLMResult) -> None:
    assert isinstance(vlm_generate_text_cpu_integration.metrics, dict)

def test_generated_text_is_not_empty(vlm_generate_text_cpu_integration: VLMResult) -> None:
    assert vlm_generate_text_cpu_integration.text.strip()

def test_generated_text_describes_image_as_document(vlm_generate_text_cpu_integration: VLMResult) -> None:
    assert "document" in vlm_generate_text_cpu_integration.text.lower()

def test_generated_text_describes_image_as_a_dedication(vlm_generate_text_cpu_integration: VLMResult) -> None:
    assert "dedication" in vlm_generate_text_cpu_integration.text.lower()


