import json

import pytest  # type: ignore[import]

from test_model_path import model_path
from src.engine.ov_genai.llm import OVGenAI_LLM
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.engine.ov_genai.tool_parse import hermes, qwen35
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelType,
    ToolCallParser,
)
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig

HERMES_MODEL_PATH = model_path("OpenVINO/Qwen3-0.6B-int8_asym-ov")
QWEN35_MODEL_PATH = model_path("OpenVINO/Qwen3.5-2B-int4_sym-ov")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

MESSAGES = [
    {"role": "user", "content": "What is the weather in Warsaw in celsius? Use the get_weather tool."},
]


class _DummyRegistry:
    async def register_unload(self, model_name: str) -> bool:
        return True


async def _generate_text(model_dir, model_name: str, tool_call_parser: ToolCallParser,
                         engine_cls, model_type: ModelType) -> str:
    if not model_dir.exists():
        pytest.skip(f"Model path not found: {model_dir}")

    load_config = ModelLoadConfig(
        model_path=str(model_dir),
        model_name=model_name,
        model_type=model_type,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
        tool_call_parser=tool_call_parser,
    )
    llm = engine_cls(load_config)
    llm.load_model(load_config)

    try:
        gen_config = OVGenAI_GenConfig(
            messages=MESSAGES,
            tools=TOOLS,
            max_tokens=256,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            stream=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        outputs = []
        async for item in llm.generate_text(gen_config):
            outputs.append(item)

        assert len(outputs) == 2
        _, text = outputs
        assert isinstance(text, str) and text.strip()
        return text
    finally:
        await llm.unload_model(_DummyRegistry(), load_config.model_name)


@pytest.mark.asyncio
async def test_hermes_tool_call_integration() -> None:
    text = await _generate_text(
        HERMES_MODEL_PATH, "integration-hermes", ToolCallParser.HERMES_PARSER,
        OVGenAI_LLM, ModelType.LLM,
    )

    _, content, tool_calls = hermes.parse_generation(text, TOOLS, enable_thinking=False)

    assert tool_calls, f"Expected hermes tool calls in output: {text!r}"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"

    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert "location" in arguments
    assert "<tool_call>" not in content


@pytest.mark.asyncio
async def test_qwen35_tool_call_integration() -> None:
    text = await _generate_text(
        QWEN35_MODEL_PATH, "integration-qwen35", ToolCallParser.QWEN35_PARSER,
        OVGenAI_VLM, ModelType.VLM,
    )

    _, content, tool_calls = qwen35.parse_generation(text, TOOLS, enable_thinking=False)

    assert tool_calls, f"Expected qwen35 tool calls in output: {text!r}"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"

    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert "location" in arguments
    assert "<tool_call>" not in content
