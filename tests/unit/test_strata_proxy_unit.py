from importlib import import_module
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest  # type: ignore[import]

from src.engine.strata_proxy import StrataProxyLLM
from src.server.model_registry import MODEL_CLASS_REGISTRY
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType


ENDPOINT = "http://strata.test:8080"
SERVED_ID = "qwen3.8-flash-next-iq3_s"


def sse(events) -> bytes:
    lines = []
    for event in events:
        if event == "[DONE]":
            lines.append("data: [DONE]")
        else:
            lines.append(f"data: {json.dumps(event)}")
    return ("\n\n".join(lines) + "\n\n").encode()


def delta(**fields):
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def make_transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=ENDPOINT,
        model_name="strata-proxy",
        model_type=ModelType.LLM,
        engine=EngineType.STRATA,
        device="strata",
    )


def make_engine(load_config: ModelLoadConfig, handler) -> StrataProxyLLM:
    engine = StrataProxyLLM(load_config, transport=make_transport(handler))
    engine.load_model(load_config)
    return engine


def ok_models_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/models"
    return httpx.Response(200, json={"object": "list", "data": [{"id": SERVED_ID}]})


# --------------------------------------------------------------------- load


def test_load_model_captures_served_model_id(load_config: ModelLoadConfig) -> None:
    engine = make_engine(load_config, ok_models_handler)

    assert engine.endpoint == ENDPOINT
    assert engine.served_model_id == SERVED_ID
    assert engine._client is not None


def test_load_model_normalizes_trailing_slash_and_v1(load_config: ModelLoadConfig) -> None:
    for raw in (ENDPOINT + "/", ENDPOINT + "/v1"):
        loader = load_config.model_copy(update={"model_path": raw})
        engine = make_engine(loader, ok_models_handler)
        assert engine.endpoint == ENDPOINT


def test_load_model_endpoint_from_runtime_config(load_config: ModelLoadConfig) -> None:
    loader = load_config.model_copy(
        update={"model_path": "placeholder", "runtime_config": {"endpoint": ENDPOINT}}
    )
    engine = make_engine(loader, ok_models_handler)
    assert engine.endpoint == ENDPOINT


def test_load_model_rejects_non_url(load_config: ModelLoadConfig) -> None:
    loader = load_config.model_copy(update={"model_path": "/models/some-ir"})
    with pytest.raises(ValueError, match="endpoint URL"):
        StrataProxyLLM(loader).load_model(loader)


def test_load_model_unreachable_raises_clear_error(load_config: ModelLoadConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(RuntimeError, match="unreachable"):
        make_engine(load_config, handler)


def test_load_model_non_200_models_raises(load_config: ModelLoadConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(RuntimeError, match="status 500"):
        make_engine(load_config, handler)


def test_load_model_no_models_served_raises(load_config: ModelLoadConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    with pytest.raises(RuntimeError, match="no models"):
        make_engine(load_config, handler)


# ---------------------------------------------------------------- streaming


def streaming_handler(captured: dict):
    events = [
        delta(role="assistant"),
        delta(reasoning_content="thinking..."),
        delta(content="Hello"),
        delta(content=" world"),
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return ok_models_handler(request)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=sse(events),
            headers={"content-type": "text/event-stream"},
        )

    return handler


@pytest.mark.asyncio
async def test_generate_stream_yields_chunks_then_metrics(load_config: ModelLoadConfig) -> None:
    captured: dict = {}
    engine = make_engine(load_config, streaming_handler(captured))

    gen_config = OVGenAI_GenConfig(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        request_id="req-stream",
        temperature=0.0,
        seed=42,
        chat_template_kwargs={"enable_thinking": False},
    )

    items = [item async for item in engine.generate_type(gen_config)]

    assert items[0] == {"chat_delta": [{"reasoning_content": "thinking..."}]}
    assert items[1] == "Hello"
    assert items[2] == " world"

    metrics = items[-1]
    assert metrics["input_token"] == USAGE["prompt_tokens"]
    assert metrics["new_token"] == USAGE["completion_tokens"]
    assert metrics["total_token"] == USAGE["total_tokens"]
    assert metrics["stream"] is True
    assert metrics["proxy"] is True
    assert metrics["endpoint"] == ENDPOINT
    assert metrics["ttft (s)"] >= 0

    body = captured["body"]
    assert body["model"] == SERVED_ID
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0
    assert body["seed"] == 42
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_generate_stream_prompt_becomes_user_message(load_config: ModelLoadConfig) -> None:
    captured: dict = {}
    engine = make_engine(load_config, streaming_handler(captured))

    gen_config = OVGenAI_GenConfig(prompt="raw prompt", stream=True)
    async for _ in engine.generate_type(gen_config):
        pass

    assert captured["body"]["messages"] == [{"role": "user", "content": "raw prompt"}]


@pytest.mark.asyncio
async def test_generate_stream_upstream_error_raises(load_config: ModelLoadConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return ok_models_handler(request)
        return httpx.Response(500, text="engine exploded")

    engine = make_engine(load_config, handler)
    gen_config = OVGenAI_GenConfig(
        messages=[{"role": "user", "content": "hi"}], stream=True, request_id="req-err"
    )

    with pytest.raises(RuntimeError, match="status 500"):
        async for _ in engine.generate_type(gen_config):
            pass


@pytest.mark.asyncio
async def test_generate_stream_input_ids_rejected(load_config: ModelLoadConfig) -> None:
    engine = make_engine(load_config, ok_models_handler)
    gen_config = OVGenAI_GenConfig(input_ids=[1, 2, 3], stream=True)

    with pytest.raises(ValueError, match="input_ids"):
        async for _ in engine.generate_type(gen_config):
            pass


# ------------------------------------------------------------- non-streaming


@pytest.mark.asyncio
async def test_generate_text_yields_metrics_then_text(load_config: ModelLoadConfig) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return ok_models_handler(request)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "full answer"},
                "finish_reason": "stop",
            }],
            "usage": USAGE,
        })

    engine = make_engine(load_config, handler)
    gen_config = OVGenAI_GenConfig(
        messages=[{"role": "user", "content": "hi"}], stream=False
    )

    items = [item async for item in engine.generate_type(gen_config)]

    assert len(items) == 2
    metrics, text = items
    assert text == "full answer"
    assert metrics["input_token"] == USAGE["prompt_tokens"]
    assert metrics["new_token"] == USAGE["completion_tokens"]
    assert metrics["stream"] is False
    assert metrics["proxy"] is True
    assert captured["body"]["stream"] is False
    assert "stream_options" not in captured["body"]


# ------------------------------------------------------------------- cancel


@pytest.mark.asyncio
async def test_cancel_closes_stream_and_returns_true(load_config: ModelLoadConfig) -> None:
    engine_holder: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return ok_models_handler(request)

        async def body():
            yield sse([delta(content="Hello")])
            # A real server streams until the client disconnects; here the
            # generator hangs until cancel() marks the request, then surfaces
            # the same StreamClosed a closed httpx stream would raise.
            engine = engine_holder["engine"]
            while "req-cancel" not in engine._cancelled:
                import asyncio
                await asyncio.sleep(0.001)
            raise httpx.StreamClosed()

        return httpx.Response(
            200, content=body(), headers={"content-type": "text/event-stream"}
        )

    engine = make_engine(load_config, handler)
    engine_holder["engine"] = engine

    gen_config = OVGenAI_GenConfig(
        messages=[{"role": "user", "content": "hi"}], stream=True, request_id="req-cancel"
    )
    stream = engine.generate_type(gen_config)

    first = await stream.__anext__()
    assert first == "Hello"
    assert "req-cancel" in engine._active_streams

    assert await engine.cancel("req-cancel") is True

    remaining = [item async for item in stream]
    assert remaining == []  # cancelled stream ends quietly, without metrics
    assert "req-cancel" not in engine._active_streams


@pytest.mark.asyncio
async def test_cancel_unknown_request_id_returns_false(load_config: ModelLoadConfig) -> None:
    engine = make_engine(load_config, ok_models_handler)
    assert await engine.cancel("no-such-request") is False


@pytest.mark.asyncio
async def test_cancel_calls_response_aclose(load_config: ModelLoadConfig) -> None:
    engine = make_engine(load_config, ok_models_handler)
    fake_response = MagicMock()
    fake_response.aclose = AsyncMock()
    engine._active_streams["req-x"] = fake_response

    assert await engine.cancel("req-x") is True
    fake_response.aclose.assert_awaited_once()
    assert "req-x" in engine._cancelled


# ------------------------------------------------------------------- unload


@pytest.mark.asyncio
async def test_unload_model_closes_client(load_config: ModelLoadConfig) -> None:
    engine = make_engine(load_config, ok_models_handler)

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    result = await engine.unload_model(registry, "strata-proxy")

    assert result is True
    assert engine._client is None
    registry.register_unload.assert_called_once_with("strata-proxy")


# ----------------------------------------------------------------- registry


def test_model_class_registry_resolves_strata_llm() -> None:
    class_path = MODEL_CLASS_REGISTRY[(EngineType.STRATA, ModelType.LLM)]
    module_path, class_name = class_path.rsplit(".", 1)
    resolved = getattr(import_module(module_path), class_name)
    assert resolved is StrataProxyLLM
