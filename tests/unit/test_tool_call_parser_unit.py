import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest  # type: ignore[import]
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import src.server.routes.openai as openai_routes
from src.engine.ov_genai.tool_parse import gemma4, hermes, qwen35
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


class _FakeRegistry:
    """Minimal registry stand-in: one record with a configurable parser."""

    def __init__(self, tool_call_parser: Optional[str]) -> None:
        class _Lock:
            async def __aenter__(self) -> "_Lock":
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        self._lock = _Lock()
        self._models = {
            "fake-id": SimpleNamespace(
                model_name="demo-model",
                tool_call_parser=tool_call_parser,
            )
        }


def _extract_sse_payloads(chunks: List[bytes]) -> List[str]:
    payloads: List[str] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                payloads.append(line[6:])
    return payloads


# ---- hermes parser unit tests ----


def test_parse_generation_supports_hermes_tool_call_tags() -> None:
    text = (
        "<tool_call>"
        '{"name":"search","arguments":{"query":"OpenVINO"}}'
        "</tool_call>"
    )

    _, _, tool_calls = hermes.parse_generation(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "OpenVINO"}


def test_parse_generation_supports_missing_closing_tag_until_eos() -> None:
    text = '<tool_call>{"name":"search","arguments":{"query":"vLLM"}}'

    _, _, tool_calls = hermes.parse_generation(text)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "vLLM"}


def test_parse_generation_rejects_plain_json_without_tool_call_tags() -> None:
    text = '{"name":"search","arguments":{"query":"legacy"}}'

    _, _, tool_calls = hermes.parse_generation(text)

    assert tool_calls is None


def test_hermes_stream_parser_incremental() -> None:
    parser = hermes.HermesStreamParser(enable_thinking=False)
    deltas: List[Dict[str, Any]] = []
    for chunk in (
        "The answer",
        " is.<tool_",
        'call>{"name":"search","arguments":{"query":"OpenArc"}}',
        "</tool_call>",
    ):
        deltas.extend(parser.feed(chunk))
    deltas.extend(parser.finish())

    content = "".join(d.get("content", "") for d in deltas)
    assert content == "The answer is."

    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 2
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(tool_deltas[1]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "OpenArc"
    }


# ---- qwen35 parser unit tests ----


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
    "The user asked for weather.\n" + qwen35.THINK_CLOSE + "\n\n"
    + qwen35.TOOL_OPEN + "\n<function=get_weather>\n"
    "<parameter=location>\nWarsaw\n</parameter>\n"
    "<parameter=unit>\ncelsius\n</parameter>\n"
    "</function>\n" + qwen35.TOOL_CLOSE + "\n"
)

QWEN_PARALLEL = (
    "Need two tools.\n" + qwen35.THINK_CLOSE + "\n\n"
    + qwen35.TOOL_OPEN + "\n<function=get_weather>\n"
    "<parameter=location>\nWarsaw\n</parameter>\n"
    "<parameter=unit>\ncelsius\n</parameter>\n"
    "</function>\n" + qwen35.TOOL_CLOSE + "\n"
    + qwen35.TOOL_OPEN + "\n<function=set_reminder>\n"
    "<parameter=task>\ncall mom\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n"
    "<parameter=urgent>\nTrue\n</parameter>\n"
    "</function>\n" + qwen35.TOOL_CLOSE + "\n"
)

QWEN_TYPED = (
    "Reminder time.\n" + qwen35.THINK_CLOSE + "\n\n"
    + qwen35.TOOL_OPEN + "\n<function=set_reminder>\n"
    "<parameter=task>\nwater the plants\n</parameter>\n"
    "<parameter=days>\n5\n</parameter>\n"
    "<parameter=urgent>\nFalse\n</parameter>\n"
    "</function>\n" + qwen35.TOOL_CLOSE + "\n"
)


def _qwen_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    return qwen35.parse_generation(text, QWEN_TOOLS)[2]


def test_parse_generation_supports_qwen_xml() -> None:
    tool_calls = _qwen_tool_calls(QWEN_SINGLE)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "Warsaw",
        "unit": "celsius",
    }


def test_parse_generation_supports_qwen_xml_parallel_and_bools() -> None:
    tool_calls = _qwen_tool_calls(QWEN_PARALLEL)

    assert tool_calls is not None
    assert [c["function"]["name"] for c in tool_calls] == [
        "get_weather",
        "set_reminder",
    ]
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "task": "call mom",
        "days": "3",
        "urgent": "True",
    }


def test_parse_generation_strips_thinking_and_xml() -> None:
    reasoning, content, tool_calls = qwen35.parse_generation(QWEN_SINGLE, QWEN_TOOLS)

    assert "weather" in reasoning
    assert qwen35.TOOL_OPEN not in content
    assert "<function=" not in content
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_parse_generation_qwen_typed_params() -> None:
    tool_calls = _qwen_tool_calls(QWEN_TYPED)
    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"task": "water the plants", "days": "5", "urgent": "False"}


def test_parse_generation_without_think_close_is_pure_content() -> None:
    reasoning, content, tool_calls = qwen35.parse_generation("just an answer", QWEN_TOOLS)
    assert reasoning == ""
    assert content == "just an answer"
    assert tool_calls is None


def _feed_tokenized(text: str, **parser_kwargs):
    """Feed text through QwenXMLToolParser with synthesized boundary IDs."""
    parser = qwen35.QwenXMLToolParser(**parser_kwargs)
    synth = qwen35._TextTagSynthesizer()
    content: List[str] = []
    for chunk, delta_tokens in synth.feed(text):
        content.append(parser.parse({}, chunk, delta_tokens))
    tail = synth.flush()
    if tail:
        content.append(parser.parse({}, tail, None))
    return parser, "".join(content)


def test_boundary_requires_token_id() -> None:
    parser = qwen35.QwenXMLToolParser()
    out = parser.parse({}, "abc" + qwen35.TOOL_OPEN + "def", None)
    assert out == "abc" + qwen35.TOOL_OPEN + "def"
    assert parser.completed_calls == []


def test_boundary_fires_on_token_id_and_slices_tag() -> None:
    parser = qwen35.QwenXMLToolParser()
    out = parser.parse({}, "abc" + qwen35.TOOL_OPEN + "def", [qwen35.TOOL_OPEN_ID])
    assert out == "abc"
    assert parser._calls_seen == 1


def test_grouped_delta_with_both_boundary_ids() -> None:
    payload = (
        qwen35.TOOL_OPEN + "\n<function=get_weather>\n"
        "<parameter=location>\nWarsaw\n</parameter>\n"
        "</function>\n" + qwen35.TOOL_CLOSE
    )
    parser = qwen35.QwenXMLToolParser()
    out = parser.parse({}, payload, [qwen35.TOOL_OPEN_ID, qwen35.TOOL_CLOSE_ID])
    assert out == ""
    assert len(parser.completed_calls) == 1
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {
        "location": "Warsaw"
    }


def test_fragment_stream_shape() -> None:
    parser, _ = _feed_tokenized(QWEN_SINGLE)
    fragments: List[Dict[str, Any]] = []
    parser2 = qwen35.QwenXMLToolParser(on_fragment=fragments.append)
    synth = qwen35._TextTagSynthesizer()
    for chunk, delta_tokens in synth.feed(QWEN_SINGLE):
        parser2.parse({}, chunk, delta_tokens)

    assert fragments[0]["function"] == {"name": "", "arguments": ""}
    assert fragments[0]["id"].startswith("call_")
    assert fragments[1]["function"] == {"name": "get_weather"}
    args = "".join(
        f["function"]["arguments"]
        for f in fragments
        if "arguments" in f.get("function", {})
    )
    assert json.loads(args) == {"location": "Warsaw", "unit": "celsius"}
    assert [f["index"] for f in fragments] == [0] * len(fragments)


def test_ramble_after_call_sets_tool_call_stop() -> None:
    parser, content = _feed_tokenized(QWEN_SINGLE + " oops extra text")
    assert parser.status == qwen35.StreamingStatus.TOOL_CALL_STOP
    assert parser.get_status() == qwen35.StreamingStatus.TOOL_CALL_STOP
    assert "oops" not in content


def test_sequential_parallel_calls_allowed_by_default() -> None:
    parser, content = _feed_tokenized(QWEN_PARALLEL)
    assert parser.status == qwen35.StreamingStatus.RUNNING
    assert [c["function"]["name"] for c in parser.completed_calls] == [
        "get_weather",
        "set_reminder",
    ]


def test_stop_after_tool_call_stops_first_call() -> None:
    parser, _ = _feed_tokenized(QWEN_PARALLEL, stop_after_tool_call=True)
    assert [c["function"]["name"] for c in parser.completed_calls] == ["get_weather"]
    assert parser.status == qwen35.StreamingStatus.TOOL_CALL_STOP


def test_malformed_block_records_error_and_valid_args() -> None:
    text = (
        qwen35.TOOL_OPEN + "\ngarbage here\n" + qwen35.TOOL_CLOSE + "\n"
    )
    parser, _ = _feed_tokenized(text)
    assert any("expected" in e for e in parser.errors)
    assert parser.completed_calls[0]["function"]["arguments"] == "{}"


def test_finalize_force_closes_unterminated_call() -> None:
    text = (
        qwen35.TOOL_OPEN + "\n<function=get_weather>\n"
        "<parameter=location>\nWarsaw\n</parameter>\n"
        "</function>\n"
    )
    parser, _ = _feed_tokenized(text)
    parser.finalize()
    assert any("unterminated" in e for e in parser.errors)
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {
        "location": "Warsaw"
    }


def test_parse_mutates_msg_with_index_keyed_calls() -> None:
    parser = qwen35.QwenXMLToolParser()
    synth = qwen35._TextTagSynthesizer()
    msg: Dict[str, Any] = {}
    for chunk, delta_tokens in synth.feed(QWEN_PARALLEL):
        parser.parse(msg, chunk, delta_tokens)
    assert msg["tool_calls"]["0"]["function"]["name"] == "get_weather"
    assert msg["tool_calls"]["1"]["function"]["name"] == "set_reminder"


def test_stream_parser_facade_split_boundary_tag() -> None:
    text = (
        "Answer. " + qwen35.TOOL_OPEN + "\n<function=get_weather>\n"
        "<parameter=location>\nOslo\n</parameter>\n"
        "</function>\n" + qwen35.TOOL_CLOSE
    )
    cut = text.index(qwen35.TOOL_OPEN) + 3  # inside the open tag
    parser = qwen35.Qwen35StreamParser(QWEN_TOOLS, enable_thinking=False)
    deltas: List[Dict[str, Any]] = []
    deltas.extend(parser.feed(text[:cut]))
    deltas.extend(parser.feed(text[cut:]))
    deltas.extend(parser.finish())

    content = "".join(d.get("content", "") for d in deltas)
    assert content == "Answer. "
    tool_frags = [f for d in deltas for f in d.get("tool_calls", [])]
    args = "".join(
        f["function"]["arguments"]
        for f in tool_frags
        if "arguments" in f.get("function", {})
    )
    assert json.loads(args) == {"location": "Oslo"}


# ---- _apply_tool_choice ----


def test_apply_tool_choice_none_hides_tools() -> None:
    messages, tools = openai_routes._apply_tool_choice(
        [{"role": "user", "content": "Weather?"}], QWEN_TOOLS, "none", None
    )
    assert messages == [{"role": "user", "content": "Weather?"}]
    assert tools is None


def test_apply_tool_choice_required_adds_instruction() -> None:
    messages, tools = openai_routes._apply_tool_choice(
        [{"role": "user", "content": "Hello"}], QWEN_TOOLS, "required", None
    )
    assert messages[0]["role"] == "system"
    assert "must emit at least one tool call" in messages[0]["content"].lower()
    assert tools == QWEN_TOOLS


def test_apply_named_tool_choice_filters_tools() -> None:
    messages, tools = openai_routes._apply_tool_choice(
        [{"role": "system", "content": "Be terse"}, {"role": "user", "content": "Do it"}],
        QWEN_TOOLS,
        {"type": "function", "function": {"name": "set_reminder"}},
        False,
    )
    assert [tool["function"]["name"] for tool in tools] == ["set_reminder"]
    assert "set_reminder" in messages[0]["content"]
    assert "at most one" in messages[0]["content"].lower()


def test_apply_named_tool_choice_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="unknown function"):
        openai_routes._apply_tool_choice(
            [{"role": "user", "content": "Do it"}],
            QWEN_TOOLS,
            {"type": "function", "function": {"name": "missing"}},
            None,
        )


# ---- route tests ----


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
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("hermes"))

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
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("hermes"))

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
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("qwen35"))

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
    # Tools + qwen35 parser -> the engine streams parsed deltas; the fake
    # worker yields the Qwen35ToolCallStreamer queue items directly.
    seen_configs: List[Any] = []

    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            seen_configs.append(generation_config)
            yield {"chat_delta": [{"reasoning_content": "thinking"}]}
            yield {
                "chat_delta": [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        ]
                    },
                    {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]},
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"location": '}},
                            {"index": 0, "function": {"arguments": '"Warsaw"}'}},
                        ]
                    },
                ]
            }
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("qwen35"))

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

    assert seen_configs and seen_configs[0].tool_call_parser == "qwen35"

    payloads = [json.loads(p) for p in _extract_sse_payloads(chunks) if p != "[DONE]"]
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "") for p in payloads
    )
    assert reasoning == "thinking"
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
    assert json.loads(args) == {"location": "Warsaw"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming_qwen35_no_tools_uses_text_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No tools -> ChunkStreamer text path; the facade does reasoning-only
    # splitting plus the tool-XML guard.
    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            text = " pondering" + qwen35.THINK_CLOSE + "Just an answer."
            for i in range(0, len(text), 5):
                yield text[i : i + 5]
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("qwen35"))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    chunks: List[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    payloads = [json.loads(p) for p in _extract_sse_payloads(chunks) if p != "[DONE]"]
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "") for p in payloads[:-1]
    )
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in payloads[:-1]
    )
    assert reasoning == " pondering"
    assert content == "Just an answer."
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_select_streamer_picks_tool_streamer_for_qwen35_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engine.ov_genai import streamers as streamers_mod

    sentinels = {"tool": object(), "chunk": object()}

    class _FakeToolStreamer:
        def __init__(self, tokenizer, gen_config):
            self.args = (tokenizer, gen_config)

    class _FakeChunkStreamer:
        def __init__(self, tokenizer, gen_config):
            self.args = (tokenizer, gen_config)

    monkeypatch.setattr(streamers_mod, "ChunkStreamer", _FakeChunkStreamer)
    monkeypatch.setattr(
        streamers_mod.qwen35_tool_parse, "Qwen35ToolCallStreamer", _FakeToolStreamer
    )

    tokenizer = object()
    tool_config = SimpleNamespace(
        tools=[{"type": "function"}], tool_call_parser="qwen35", chat_template_kwargs={}
    )
    picked = streamers_mod.select_streamer(tokenizer, tool_config)
    assert isinstance(picked, _FakeToolStreamer)
    assert picked.args[0] is tokenizer

    plain_config = SimpleNamespace(tools=None, tool_call_parser="qwen35", chat_template_kwargs={})
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, plain_config), _FakeChunkStreamer
    )

    hermes_config = SimpleNamespace(
        tools=[{"type": "function"}], tool_call_parser="hermes", chat_template_kwargs={}
    )
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, hermes_config), _FakeChunkStreamer
    )


# ---- unset tool_call_parser ----


@pytest.mark.asyncio
async def test_tools_request_rejected_without_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Workers:
        async def generate(self, model_name: str, generation_config: Any) -> Dict[str, Any]:
            return {"text": "hello", "metrics": {}}

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry(None))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Hi"}],
        tools=QWEN_TOOLS,
        stream=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await openai_routes.openai_chat_completions(request, _DummyRequest())
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_plain_request_passthrough_without_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "Plain answer. <tool_call> not parsed."

    class _Workers:
        async def generate(self, model_name: str, generation_config: Any) -> Dict[str, Any]:
            return {
                "text": raw_text,
                "metrics": {"input_token": 2, "new_token": 3, "total_token": 5},
            }

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry(None))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == raw_text
    assert "reasoning_content" not in choice["message"]


@pytest.mark.asyncio
async def test_streaming_passthrough_without_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            yield "chunk one. "
            yield "chunk two."
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry(None))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    chunks: List[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    payloads = [json.loads(p) for p in _extract_sse_payloads(chunks) if p != "[DONE]"]
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in payloads[:-1]
    )
    assert content == "chunk one. chunk two."
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


# ---- gemma4 tool-call parser ----


GEMMA_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]


def _feed_gemma(spec, **parser_kwargs):
    """Drive Gemma4ToolCallParser with driver-contract deltas.

    spec: sequence of (delta_text, delta_tokens) pairs; text deltas never
    carry protocol ids (the streamer flushes text before boundary tokens).
    """
    parser = gemma4.Gemma4ToolCallParser(**parser_kwargs)
    content: List[str] = []
    for text, ids in spec:
        content.append(parser.parse({}, text, ids))
    return parser, "".join(content)


def test_gemma_boundary_requires_token_id() -> None:
    # Payload text without the open-boundary ID is plain content.
    parser, content = _feed_gemma([("call:get_weather{city:Paris}", None)])
    assert content == "call:get_weather{city:Paris}"
    assert parser.completed_calls == []


def test_gemma_boundary_fires_on_token_id() -> None:
    parser, content = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Paris}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ])
    assert content == ""
    assert len(parser.completed_calls) == 1
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {
        "city": "Paris"
    }


def test_gemma_fragment_stream_shape() -> None:
    fragments: List[Dict[str, Any]] = []
    parser = gemma4.Gemma4ToolCallParser(on_fragment=fragments.append)
    for text, ids in [
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{", None),
        ("city:Paris", None),
        (", ", None),
        ("unit:celsius", None),
        ("}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ]:
        parser.parse({}, text, ids)

    assert fragments[0]["function"] == {"name": "", "arguments": ""}
    assert fragments[0]["id"].startswith("call_")
    assert fragments[1]["function"] == {"name": "get_weather"}
    args = "".join(
        f["function"]["arguments"]
        for f in fragments
        if "arguments" in f.get("function", {})
    )
    assert json.loads(args) == {"city": "Paris", "unit": "celsius"}
    assert [f["index"] for f in fragments] == [0] * len(fragments)


def test_gemma_sequential_parallel_calls_allowed_by_default() -> None:
    parser, content = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Tokyo}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
        ("\n\n", None),
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Paris}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ])
    assert parser.status == gemma4.StreamingStatus.RUNNING
    assert [c["function"]["name"] for c in parser.completed_calls] == [
        "get_weather", "get_weather",
    ]
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert json.loads(parser.completed_calls[1]["function"]["arguments"]) == {"city": "Paris"}
    assert content == ""


def test_gemma_ramble_after_call_sets_tool_call_stop() -> None:
    parser, content = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Paris}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
        (" oops extra text", None),
    ])
    assert parser.status == gemma4.StreamingStatus.TOOL_CALL_STOP
    assert parser.get_status() == gemma4.StreamingStatus.TOOL_CALL_STOP
    assert "oops" not in content


def test_gemma_stop_after_tool_call_stops_first_call() -> None:
    parser, _ = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:first{a:1}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:second{b:2}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ], stop_after_tool_call=True)
    assert [c["function"]["name"] for c in parser.completed_calls] == ["first"]
    assert parser.status == gemma4.StreamingStatus.TOOL_CALL_STOP


def test_gemma_malformed_payload_records_error_and_valid_args() -> None:
    parser, _ = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("garbage here", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ])
    assert any("expected 'call:'" in e for e in parser.errors)
    assert parser.completed_calls[0]["function"]["arguments"] == "{}"


def test_gemma_finalize_force_closes_unterminated_call() -> None:
    parser, _ = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Oslo", None),
    ])
    parser.finalize()
    assert any("unterminated" in e for e in parser.errors)
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {
        "city": "Oslo"
    }


def test_gemma_parse_mutates_msg_with_index_keyed_calls() -> None:
    parser = gemma4.Gemma4ToolCallParser()
    msg: Dict[str, Any] = {}
    for text, ids in [
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:get_weather{city:Tokyo}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:set_alarm{when:7am}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ]:
        parser.parse(msg, text, ids)
    assert msg["tool_calls"]["0"]["function"]["name"] == "get_weather"
    assert msg["tool_calls"]["1"]["function"]["name"] == "set_alarm"


def test_gemma_nested_brace_value_and_zero_args() -> None:
    parser, _ = _feed_gemma([
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:foo{opts:{a:1, b:{c:2}}, note:hi}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
        ("", [gemma4.TOOL_OPEN_ID]),
        ("call:noargs{}", None),
        ("", [gemma4.TOOL_CLOSE_ID]),
    ])
    assert json.loads(parser.completed_calls[0]["function"]["arguments"]) == {
        "opts": "{a:1, b:{c:2}}",
        "note": "hi",
    }
    assert parser.completed_calls[1]["function"]["arguments"] == "{}"


def test_gemma_channel_splitter_thought_channel() -> None:
    splitter = gemma4.Gemma4ChannelSplitter()
    deltas = [
        ("Hello ", []),
        ("", [gemma4.CHANNEL_OPEN_ID]),
        ("thought\nLet me think.\nMore thought.\n", []),
        ("", [gemma4.CHANNEL_CLOSE_ID]),
        ("Final answer", []),
    ]
    reason, content = "", ""
    for text, ids in deltas:
        r, c, passthrough = splitter.feed(text, ids)
        reason += r
        content += c
        assert passthrough == [] or all(i not in (100, 101) for i in passthrough)
    assert reason == "Let me think.\nMore thought."
    assert content == "Hello Final answer"


def test_gemma_channel_splitter_opaque_channel_passthrough() -> None:
    splitter = gemma4.Gemma4ChannelSplitter()
    deltas = [
        ("", [gemma4.CHANNEL_OPEN_ID]),
        ("summary\nstuff\n", []),
        ("", [gemma4.CHANNEL_CLOSE_ID]),
        ("tail", []),
    ]
    reason, content = "", ""
    for text, ids in deltas:
        r, c, _ = splitter.feed(text, ids)
        reason += r
        content += c
    assert reason == ""
    assert content == "summary\nstuff\ntail"


def test_gemma_channel_splitter_header_across_deltas() -> None:
    splitter = gemma4.Gemma4ChannelSplitter()
    r1, c1, _ = splitter.feed("", [gemma4.CHANNEL_OPEN_ID])
    r2, c2, _ = splitter.feed("thou", [])          # header split mid-name
    r3, c3, _ = splitter.feed("ght\nbody", [])
    r4, c4, _ = splitter.feed("", [gemma4.CHANNEL_CLOSE_ID])
    assert "".join([r1, r2, r3, r4]) == "body"
    assert "".join([c1, c2, c3, c4]) == ""


def test_gemma_parse_generation_reasoning_and_parallel_calls() -> None:
    raw = (
        "<|channel>thought\nstep one\nstep two\n\n<channel|>\nHello\n\n"
        "<|tool_call>call:get_weather{city:Tokyo}<tool_call|>\n"
        "<|tool_call>call:get_weather{city:Paris, when:today}<tool_call|><turn|>"
    )
    reasoning, content, calls = gemma4.parse_generation(raw)
    assert reasoning == "step one\nstep two"
    assert content == "Hello"
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_weather"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"city": "Paris", "when": "today"}
    assert all(c["id"].startswith("call_") for c in calls)


def test_gemma_parse_generation_passthrough_without_tags() -> None:
    assert gemma4.parse_generation("plain answer") == ("", "plain answer", None)


def test_gemma_parse_generation_unterminated_tool_block() -> None:
    raw = "Intro <|tool_call>call:get_weather{city:Oslo}"
    reasoning, content, calls = gemma4.parse_generation(raw)
    assert reasoning == ""
    assert calls is None
    assert "call:get_weather{city:Oslo}" in content


def test_gemma_wants_engine_stream() -> None:
    assert gemma4.wants_engine_stream([{"type": "function"}], False) is True
    assert gemma4.wants_engine_stream(None, True) is True
    assert gemma4.wants_engine_stream(None, False) is False


def test_select_streamer_gemma4_engine_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.engine.ov_genai import streamers as streamers_mod

    class _FakeGemmaStreamer:
        def __init__(self, tokenizer, gen_config):
            self.args = (tokenizer, gen_config)

    class _FakeChunkStreamer:
        def __init__(self, tokenizer, gen_config):
            self.args = (tokenizer, gen_config)

    monkeypatch.setattr(streamers_mod, "ChunkStreamer", _FakeChunkStreamer)
    monkeypatch.setattr(
        streamers_mod.gemma4_tool_parse, "Gemma4ToolCallStreamer", _FakeGemmaStreamer
    )

    tokenizer = object()
    # Tools present -> engine streamer even with thinking off.
    tools_cfg = SimpleNamespace(
        tools=[{"type": "function"}],
        tool_call_parser="gemma4",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, tools_cfg), _FakeGemmaStreamer
    )

    # Thinking on, no tools -> engine streamer (channel tags are ID-only).
    think_cfg = SimpleNamespace(
        tools=None,
        tool_call_parser="gemma4",
        chat_template_kwargs={"enable_thinking": True},
    )
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, think_cfg), _FakeGemmaStreamer
    )

    # Default (no chat_template_kwargs) counts as thinking enabled.
    default_cfg = SimpleNamespace(
        tools=None, tool_call_parser="gemma4", chat_template_kwargs={}
    )
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, default_cfg), _FakeGemmaStreamer
    )

    # Thinking explicitly off, no tools -> plain ChunkStreamer.
    off_cfg = SimpleNamespace(
        tools=None,
        tool_call_parser="gemma4",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert isinstance(
        streamers_mod.select_streamer(tokenizer, off_cfg), _FakeChunkStreamer
    )


# ---- gemma4 routes ----


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming_gemma4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        "<|channel>thought\npondering\n<channel|>\n"
        "<|tool_call>call:get_weather{city:Warsaw}<tool_call|><turn|>"
    )

    class _Workers:
        async def generate(self, model_name: str, generation_config: Any) -> Dict[str, Any]:
            return {
                "text": raw,
                "metrics": {"input_token": 4, "new_token": 6, "total_token": 10},
            }

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("gemma4"))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Weather in Warsaw?"}],
        tools=GEMMA_TOOLS,
        stream=False,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Warsaw"
    }
    assert choice["message"].get("reasoning_content") == "pondering"
    assert choice["message"].get("content") in (None, "")


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming_gemma4_engine_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tools + gemma4 parser -> the engine streams parsed deltas via chat_delta.
    seen_configs: List[Any] = []

    class _Workers:
        async def stream_generate(self, model_name: str, generation_config: Any) -> AsyncIterator[Any]:
            seen_configs.append(generation_config)
            yield {"chat_delta": [{"reasoning_content": "pondering"}]}
            yield {
                "chat_delta": [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        ]
                    },
                    {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]},
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city": "'}},
                            {"index": 0, "function": {"arguments": 'Warsaw"}'}},
                        ]
                    },
                ]
            }
            yield {"metrics": {"input_token": 2, "new_token": 3, "total_token": 5}}

        async def infer_cancel(self, request_id: str) -> None:
            return None

    monkeypatch.setattr(openai_routes, "_workers", _Workers())
    monkeypatch.setattr(openai_routes, "_registry", _FakeRegistry("gemma4"))

    request = OpenAIChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "Weather in Warsaw?"}],
        tools=GEMMA_TOOLS,
        stream=True,
    )

    response = await openai_routes.openai_chat_completions(request, _DummyRequest())
    chunks: List[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert seen_configs and seen_configs[0].tool_call_parser == "gemma4"

    payloads = [json.loads(p) for p in _extract_sse_payloads(chunks) if p != "[DONE]"]
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content", "") for p in payloads
    )
    assert reasoning == "pondering"
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
    assert json.loads(args) == {"city": "Warsaw"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
