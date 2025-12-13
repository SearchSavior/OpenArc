import asyncio
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig


MODEL_PATH = Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-3B-Instruct-int4_sym-ov")
UNIT_TEST_PATH = Path(__file__).with_name("test_ov_genai_vlm_unit.py")
PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

_UNIT_TESTS_PASSED: bool | None = None
_UNIT_TEST_OUTPUT: str = ""


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
            "Skipping VLM integration test because unit tests failed:\n" + _UNIT_TEST_OUTPUT
        )


def test_vlm_generate_text_cpu_integration() -> None:
    _ensure_unit_tests_pass()
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
        base64_image = f"data:image/png;base64,{PIXEL_PNG_BASE64}"

        gen_config = OVGenAI_GenConfig(
            messages=_make_messages(base64_image),
            max_tokens=16,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            stream=False,
        )

        async def _run_test():
            outputs = []
            async for item in vlm.generate_text(gen_config):
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
        asyncio.run(vlm.unload_model(_DummyRegistry(), load_config.model_name))
