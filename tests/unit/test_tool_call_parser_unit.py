import json
from typing import Any, AsyncIterator, Dict, List

import pytest  # type: ignore[import]
from fastapi.responses import StreamingResponse

import src.server.routes.openai as openai_routes
from src.server.schemas.requests_openai import OpenAIChatCompletionRequest
from src.server.utils.chat import flatten_messages, normalize_tool_calls_for_template


def test_normalize_tool_call_arguments_json_string_to_mapping() -> None:
    calls = normalize_tool_calls_for_template(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Warsaw", "unit": "celsius"}',
                },
            }
        ]
    )
    assert calls[0]["function"]["arguments"] == {
        "location": "Warsaw",
        "unit": "celsius",
    }


def test_flatten_messages_preserves_tool_role_and_parses_args() -> None:
    msgs = flatten_messages(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 22}'},
        ]
    )
    assert msgs[1]["content"] == ""
    assert msgs[1]["tool_calls"][0]["function"]["arguments"]["location"] == "Tokyo"
    assert msgs[2]["content"] == '{"temp": 22}'
    assert msgs[2]["tool_call_id"] == "call_1"



class _DummyRequest:
    async def is_disconnected(self) -> bool:
        return False


def _extract_sse_payloads(chunks: List[bytes]) -> List[str]:
    payloads: List[str] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                payloads.append(line[6:])
    return payloads


def test_parse_tool_calls_supports_hermes_tool_call_tags() -> None:
    text = (
        "<tool_call>"
        '{"name":"search","arguments":{"query":"OpenVINO"}}'
        "</tool_call>"
    )

    tool_calls = openai_routes.parse_tool_calls(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "OpenVINO"}


def test_parse_tool_calls_supports_missing_closing_tag_until_eos() -> None:
    text = '<tool_call>{"name":"search","arguments":{"query":"vLLM"}}'

    tool_calls = openai_routes.parse_tool_calls(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "vLLM"}


def test_parse_tool_calls_rejects_plain_json_without_tool_call_tags() -> None:
    text = '{"name":"search","arguments":{"query":"legacy"}}'

    tool_calls = openai_routes.parse_tool_calls(text)

    assert tool_calls is None


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Workers:
        async def generate(self, model_name: str, generation_config: Any) -> Dict[str, Any]:
            return {
                "text": (
                    "<tool_call>"
                    '{"name":"search","arguments":{"query":"OpenArc"}}'
                    "</tool_call>"
                ),
                "metrics": {"input_token": 4, "new_token": 6, "total_token": 10},
            }

    monkeypatch.setattr(openai_routes, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Find OpenArc docs"}],
        stream=False,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert len(choice["message"]["tool_calls"]) == 1
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "OpenArc"
    }


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming_hermes_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            yield "<tool_"
            yield 'call>{"name":"search","arguments":{"query":"OpenArc"}}'
            yield "</tool_call>"
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Find OpenArc docs"}],
        stream=True,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    assert isinstance(response, StreamingResponse)

    chunks: List[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    payloads = _extract_sse_payloads(chunks)
    assert payloads[-1] == "[DONE]"

    json_payloads = [json.loads(p) for p in payloads if p != "[DONE]"]

    content_deltas = [
        payload
        for payload in json_payloads
        if payload["choices"][0]["delta"].get("content")
    ]
    assert content_deltas == []

    tool_deltas = [
        payload
        for payload in json_payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_deltas) >= 2
    assert tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(
        tool_deltas[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    ) == {"query": "OpenArc"}

    assert json_payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


QWEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "days": {"type": "integer"},
                    "urgent": {"type": "boolean"},
                },
                "required": ["task", "days"],
            },
        },
    },
]


QWEN_SINGLE = (
    "The user asked for weather.\n</think>\n\n"
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=location>\nWarsaw\n</parameter>\n"
    "<parameter=unit>\ncelsius\n</parameter>\n"
    "</function>\n</tool_call>\n"
)

QWEN_PARALLEL = (
    "Need two tools.\n</think>\n\n"
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=location>\nWarsaw\n</parameter>\n"
    "<parameter=unit>\ncelsius\n</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=set_reminder>\n"
    "<parameter=task>\ncall mom\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n"
    "<parameter=urgent>\nTrue\n</parameter>\n"
    "</function>\n</tool_call>\n"
)

QWEN_TYPED = (
    "Reminder time.\n</think>\n\n"
    "<tool_call>\n<function=set_reminder>\n"
    "<parameter=task>\nwater the plants\n</parameter>\n"
    "<parameter=days>\n5\n</parameter>\n"
    "<parameter=urgent>\nFalse\n</parameter>\n"
    "</function>\n</tool_call>\n"
)


def test_parse_tool_calls_supports_qwen_xml() -> None:
    tool_calls = openai_routes.parse_tool_calls(QWEN_SINGLE, QWEN_TOOLS)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "Warsaw",
        "unit": "celsius",
    }


def test_parse_tool_calls_supports_qwen_xml_parallel_and_bools() -> None:
    tool_calls = openai_routes.parse_tool_calls(QWEN_PARALLEL, QWEN_TOOLS)

    assert tool_calls is not None
    assert [c["function"]["name"] for c in tool_calls] == [
        "get_weather",
        "set_reminder",
    ]
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "task": "call mom",
        "days": 3,
        "urgent": True,
    }


def test_parse_generation_strips_thinking_and_xml() -> None:
    reasoning, content, tool_calls = openai_routes.parse_generation(
        QWEN_SINGLE, QWEN_TOOLS
    )

    assert "weather" in reasoning
    assert "<tool_call>" not in content
    assert "<function=" not in content
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_parse_tool_calls_qwen_typed_params() -> None:
    tool_calls = openai_routes.parse_tool_calls(QWEN_TYPED, QWEN_TOOLS)
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["days"] == 5
    assert args["urgent"] is False


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming_qwen_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Workers:
        async def generate(self, model_name: str, generation_config: Any) -> Dict[str, Any]:
            return {
                "text": QWEN_SINGLE,
                "metrics": {"input_token": 4, "new_token": 6, "total_token": 10},
            }

    monkeypatch.setattr(openai_routes, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Weather in Warsaw?"}],
        tools=QWEN_TOOLS,
        stream=False,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "reasoning_content" in choice["message"]
    assert choice["message"].get("content") in (None, "", "\n\n")


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming_qwen_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            text = QWEN_SINGLE
            for i in range(0, len(text), 7):
                yield text[i : i + 7]
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Weather in Warsaw?"}],
        tools=QWEN_TOOLS,
        stream=True,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    chunks: List[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    payloads = [json.loads(p) for p in _extract_sse_payloads(chunks) if p != "[DONE]"]
    names = []
    args = ""
    for payload in payloads:
        for frag in payload["choices"][0]["delta"].get("tool_calls") or []:
            fn = frag.get("function") or {}
            if fn.get("name"):
                names.append(fn["name"])
            if fn.get("arguments"):
                args += fn["arguments"]
    assert "get_weather" in names
    assert json.loads(args) == {"location": "Warsaw", "unit": "celsius"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
