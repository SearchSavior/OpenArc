"""Assert the whole request/response chain logs the right thing when a worker
fails. A worker the supervisor is still respawning within its respawn budget
(``RemoteWorkerDeadError.will_respawn==True``) is a *healing* event, so its
in-flight request must log a single short, cause-only line -- never a full
traceback / "Unhandled exception" / uvicorn's "Exception in ASGI application".
The companion test makes the death terminal (``will_respawn==False``, budget
exhausted) and asserts the opposite: a full traceback, no short line.

Drives a real streaming request through the real FastAPI app + the real
``/v1/chat/completions`` and ``/v1/completions`` routes via an injected model
that reports the death (heal vs terminal, chosen by a request marker on the real
``will_respawn`` flag). Run with: pytest -W ignore <this file>.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping

import httpx
import pytest  # type: ignore[import]
from fastapi import FastAPI

import src.server.deps as deps_module
import src.server.model_registry as model_registry_module
import src.server.routes.openai as openai_module  # noqa: F401 (registers the routes)
import src.server.worker_registry as worker_module
from src.engine.worker.protocol import RemoteWorkerDeadError
from src.server.main import global_exception_handler
from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType
from src.server.routes.openai import router as openai_router

# Prompt/message marker: one injected model serves both the heal and terminal
# cases (it sets will_respawn on the RemoteWorkerDeadError it raises).
HEAL_MARKER = "OPENARC_WORKER_HEAL"          # supervisor still respawning
TERMINAL_MARKER = "OPENARC_WORKER_TERMINAL"  # respawn budget exhausted

# The loggers under test (filtering to these keeps unrelated logs out of the assertions).
SERVER_LOGGERS = ("src.server.worker_registry", "src.server.routes.openai", "src.server.main")

# Marks that classify a log line as loud (terminal) vs quiet (healing).
FULL_TRACE_MARKERS = ("Unhandled exception", "Full traceback")
ROUTE_GRACEFUL_MARKERS = ("worker restart in progress", "ending stream for")
REGISTRY_HEAL_MARKERS = (
    "being restarted within its respawn budget",   # _log_inference_failure (heal)
    "restarting it within budget",                 # _commit_completed_packet (heal)
)

# Each route differs only in the SSE object name + terminal choice shape (what the helper unifies).
CHAT_CASE = {
    "id": "chat",
    "path": "/v1/chat/completions",
    "object": "chat.completion.chunk",
    "choice": {"index": 0, "delta": {}, "finish_reason": "error"},
    "body": lambda model: {
        "model": model,
        "messages": [{"role": "user", "content": ""}],  # marker filled in per case
        "stream": True,
        "max_tokens": 8,
    },
}
COMPLETIONS_CASE = {
    "id": "completions",
    "path": "/v1/completions",
    "object": "text_completion.chunk",
    "choice": {"index": 0, "text": "", "finish_reason": "error"},
    "body": lambda model: {"model": model, "prompt": "", "stream": True, "max_tokens": 8},
}
STREAM_CASES = [CHAT_CASE, COMPLETIONS_CASE]


class _WedgeLLM:
    """A model whose ``generate_type`` raises a ``RemoteWorkerDeadError`` for a
    request carrying a marker, and serves genuine (tiny) output otherwise.

    Being a ``worker_module.OVGenAI_LLM`` (which the test monkeypatches to *this*
    class) it routes through the very same ``infer_llm`` and ``stream_generate``
    path the production engine uses; the only test-controlled behaviour is the
    will_respawn flag it stamps on the failure.
    """

    def __init__(self, load_config: Any) -> None:
        self.load_config = load_config

    async def unload_model(self, registry: Any, model_name: str) -> bool:
        return True

    async def generate_type(self, gen_config: Any):
        text = gen_config.prompt or ""
        for msg in gen_config.messages or []:
            content = msg.get("content") if isinstance(msg, Mapping) else None
            if isinstance(content, str):
                text += " " + content
        if TERMINAL_MARKER in text:
            raise RemoteWorkerDeadError(
                "OV device wedged (CL_OUT_OF_RESOURCES); respawn budget exhausted",
                original_type="RuntimeError",
                will_respawn=False,
            )
        if HEAL_MARKER in text:
            raise RemoteWorkerDeadError(
                "OV device wedged (CL_OUT_OF_RESOURCES); supervisor respawning",
                original_type="RuntimeError",
                will_respawn=True,
            )
        # Healthy path: a metrics dict, then one text token.
        yield {"new_token": 2, "stream": False, "load_time (s)": 0.0}
        yield "ok"


async def _fake_create_model_instance(load_config: Any) -> Any:
    return _WedgeLLM(load_config)


def _load_config(model_name: str) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path="/does/not/need/to/exist",
        model_name=model_name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )


def _build_app() -> FastAPI:
    """A minimal app that runs the real OpenAI routes with the real global ASGI
    exception handler -- the same handler the production ``src.server.main``
    installs -- so an escaping exception is logged exactly as it would be in
    production (short line vs full "Unhandled exception" + "Full traceback")."""
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    app.include_router(openai_router)
    return app


async def _drive(monkeypatch, case: dict, model_name: str, marker: str) -> dict:
    """Drive one streaming request through a fresh, real app. Returns the
    outcome as plain values so the (non-async) test can assert, plus whatever
    the response body held. A terminal failure re-raises out of the ASGI
    transport (Starlette re-raises after the response started), so the raise is
    captured here -- the logs, which is what the test actually asserts, have
    already fired during the call."""
    model_registry = ModelRegistry()
    worker_registry = worker_module.WorkerRegistry(model_registry)
    monkeypatch.setattr(worker_module, "OVGenAI_LLM", _WedgeLLM, raising=False)
    monkeypatch.setattr(
        model_registry_module,
        "create_model_instance",
        _fake_create_model_instance,
        raising=False,
    )
    monkeypatch.setattr(openai_module, "_workers", worker_registry, raising=False)
    monkeypatch.setattr(openai_module, "_registry", model_registry, raising=False)
    # Auth is off by default; force it in case a developer's shell exports
    # OPENARC_API_KEY_REQUIRED (the module global is otherwise frozen at import).
    monkeypatch.setattr(deps_module, "AUTH_REQUIRED", False, raising=False)

    outcome: dict = {"raised": None, "status": None, "body": ""}
    await model_registry.register_load(_load_config(model_name))
    await asyncio.sleep(0)  # let the worker task start consuming the queue
    try:
        body = case["body"](model_name)
        body = _inject_marker(body, marker)
        transport = httpx.ASGITransport(app=_build_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as cli:
            try:
                resp = await cli.post(case["path"], json=body)
            except Exception as exc:  # mid-stream terminal failure re-raises
                outcome["raised"] = repr(exc)
            else:
                outcome["status"] = resp.status_code
                outcome["body"] = resp.text
    finally:
        # Stop the worker task so it does not outlive this event loop.
        await model_registry.register_unload(model_name)
        await asyncio.sleep(0)
    return outcome


def _inject_marker(body: dict, marker: str) -> dict:
    """Put the control marker into the request's prompt messages."""
    if "messages" in body:
        body = dict(body)
        messages = [dict(m, content=(str(m.get("content") or "") + f" {marker}").strip())
                    for m in body["messages"]]
        body["messages"] = messages
        return body
    if "prompt" in body:
        body = dict(body, prompt=f"{marker} {body['prompt']}".strip())
    return body


def _server_logs(caplog: Any) -> list:
    return [r for r in caplog.records if r.name in SERVER_LOGGERS]


def _logged_lines(caplog: Any) -> list:
    return [r.getMessage() for r in _server_logs(caplog)]


def _has_full_trace(caplog: Any) -> bool:
    """True when a full traceback was logged -- either an error record carrying
    an ``exc_info`` traceback (the registry's ``_log_inference_failure`` /
    ``_commit_completed_packet`` loud branch), or the global ASGI handler's own
    ``Unhandled exception`` / ``Full traceback`` lines. This is the signal that
    the exception was NOT kept on the quiet heal path."""
    if any(r.levelno >= logging.ERROR and r.exc_info for r in _server_logs(caplog)):
        return True
    return any(any(m in line for m in FULL_TRACE_MARKERS) for line in _logged_lines(caplog))


def _has_route_graceful(caplog: Any) -> bool:
    """True when the route took the graceful path and emitted the short line the
    extracted ``_end_stream_on_worker_restart`` helper writes."""
    for r in caplog.records:
        if r.name == "src.server.routes.openai":
            line = r.getMessage()
            if all(m in line for m in ROUTE_GRACEFUL_MARKERS):
                return True
    return False


def _has_registry_heal_line(caplog: Any) -> bool:
    return any(
        any(m in r.getMessage() for m in REGISTRY_HEAL_MARKERS) for r in _server_logs(caplog)
    )


def _sse_chunks(body: str) -> list:
    """Parse an SSE body into its ``data:`` payloads: JSON dicts for the real
    chunks and the literal ``"[DONE]"`` for the terminator."""
    chunks: list = []
    for segment in body.split("\n\n"):
        segment = segment.strip()
        if not segment:
            continue
        if not segment.startswith("data:"):
            continue
        payload = segment[len("data:"):].strip()
        if payload == "[DONE]":
            chunks.append("[DONE]")
        else:
            chunks.append(json.loads(payload))
    return chunks


# --------------------------------------------------------------------------- #
# Healing: a worker being respawned within its budget must stay quiet and the
# stream must close gracefully. Exercised with the extracted helper in BOTH
# stream routes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", STREAM_CASES, ids=lambda c: c["id"])
def test_streamed_request_on_worker_respawn_is_quiet_and_graceful(
    case: dict, monkeypatch: "pytest.MonkeyPatch", caplog: Any,
) -> None:
    caplog.set_level(logging.INFO)

    outcome = asyncio.run(
        _drive(monkeypatch, case, f"wedge-{case['id']}", HEAL_MARKER)
    )

    # The worker is merely being respawned: no full traceback anywhere.
    assert not _has_full_trace(caplog), _logged_lines(caplog)
    # ...and the quiet line proves it: the route ended the stream courteously
    # (via _end_stream_on_worker_restart) instead of letting it escape.
    assert _has_route_graceful(caplog), _logged_lines(caplog)
    assert _has_registry_heal_line(caplog), _logged_lines(caplog)

    # The response actually closed cleanly: 200, the graceful terminal chunk,
    # then [DONE] -- nothing raised to the ASGI handler.
    assert outcome["raised"] is None, outcome
    assert outcome["status"] == 200
    chunks = _sse_chunks(outcome["body"])
    assert "[DONE]" in chunks
    terminal = [
        c for c in chunks
        if isinstance(c, dict)
        and c.get("object") == case["object"]
        and c.get("choices")
        and c["choices"][0].get("finish_reason") == "error"
    ]
    assert len(terminal) == 1, chunks
    assert terminal[0]["choices"][0] == case["choice"], terminal[0]["choices"][0]


# --------------------------------------------------------------------------- #
# Terminal: the respawn budget is exhausted, so the full traceback SHOULD appear
# -- this is the behaviour the dedup must not accidentally silence.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", STREAM_CASES, ids=lambda c: c["id"])
def test_streamed_request_on_terminal_worker_failure_logs_full_traceback(
    case: dict, monkeypatch: "pytest.MonkeyPatch", caplog: Any,
) -> None:
    caplog.set_level(logging.INFO)

    outcome = asyncio.run(
        _drive(monkeypatch, case, f"dead-{case['id']}", TERMINAL_MARKER)
    )

    # The failure is terminal: a full traceback is logged (loudness preserved).
    assert _has_full_trace(caplog), _logged_lines(caplog)
    # And it did NOT take the graceful path -- the extracted helper's short line
    # must be absent, proving terminal failures are genuinely loud.
    assert not _has_route_graceful(caplog), _logged_lines(caplog)
    # A terminal streaming failure surfaces to the client as an error (either a
    # 500 or a re-raised exception, depending on whether the stream had started),
    # but never the graceful 200 from the heal path.
    assert outcome["raised"] is not None or outcome["status"] != 200, outcome
