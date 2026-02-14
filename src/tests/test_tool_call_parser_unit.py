import json
from typing import Any, AsyncIterator, Dict, List

import pytest  # type: ignore[import]
from fastapi.responses import StreamingResponse

import src.server.main as server_main
from src.server.models.requests_openai import OpenAIChatCompletionRequest


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

    tool_calls = server_main.parse_tool_calls(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "OpenVINO"}


def test_parse_tool_calls_supports_missing_closing_tag_until_eos() -> None:
    text = '<tool_call>{"name":"search","arguments":{"query":"vLLM"}}'

    tool_calls = server_main.parse_tool_calls(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "vLLM"}


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

    monkeypatch.setattr(server_main, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Find OpenArc docs"}],
        stream=False,
    )

    response = await server_main.openai_chat_completions(request, _DummyRequest())

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

    monkeypatch.setattr(server_main, "_workers", _Workers())

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Find OpenArc docs"}],
        stream=True,
    )

    response = await server_main.openai_chat_completions(request, _DummyRequest())
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
