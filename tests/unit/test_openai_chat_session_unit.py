"""Route-level tests for /v1/chat/completions session handling.

These drive the real ``openai_chat_completions`` handler with a fake
``_workers`` and `_registry`, and a real ``SessionRegistry``, to prove the three
guarantees end to end:

  * with ``--sih`` off / no session header -> behaviour is unchanged;
  * the session's accumulated "current context" is what the client reads
    (``usage.total_tokens``; goose shows that value before / context_window);
  * a worker restart (crash, will_respawn) keeps reporting the session's
    last-known value instead of 0, and a fresh run re-bases it.
"""

import asyncio
import json
import types
from typing import Any, Optional

import pytest  # type: ignore[import]
from fastapi import HTTPException, Request

import src.server.routes.openai as R
from src.server.sessions import SessionRegistry
from src.server.schemas.requests_openai import OpenAIChatCompletionRequest
from src.engine.worker import protocol as proto

HDR = "X-Session-Id"
MODEL = "m1"


def _record(tool=None) -> Any:
    return types.SimpleNamespace(model_name=MODEL, tool_call_parser=tool, model_config_blocks=None)


def _registry(**models: Any) -> Any:
    return types.SimpleNamespace(
        _lock=asyncio.Lock(),
        _models={name: _record(**kw) for name, kw in models.items()},
    )


def _request(hdr_value: Optional[str]) -> Request:
    headers = [] if not hdr_value else [(HDR.lower().encode(), hdr_value.encode())]

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 80),
            "path": "/v1/chat/completions",
            "query_string": "",
            "headers": headers,
        },
        # `receive` returns http.request so is_disconnected() is False during streaming.
        receive=_receive,
    )


class _Workers:
    def __init__(
        self,
        *,
        gen_result=None,
        stream_items=None,
        raise_dead: bool = False,
        will_respawn: bool = True,
    ):
        self.gen_result = gen_result
        self.stream_items = stream_items or []
        self.raise_dead = raise_dead
        self.will_respawn = will_respawn
        self.last_gen_config: Optional[Any] = None
        self.cancelled: list = []

    async def generate(self, model_name, gen_config):
        self.last_gen_config = gen_config
        if self.raise_dead:
            raise proto.RemoteWorkerDeadError("worker died", will_respawn=self.will_respawn)
        return self.gen_result

    async def stream_generate(self, model_name, gen_config):
        self.last_gen_config = gen_config
        for item in self.stream_items:
            yield item

    async def infer_cancel(self, request_id):
        self.cancelled.append(request_id)


def _plain_metrics(input_token: int = 120, new_token: int = 4) -> dict:
    return {"input_token": input_token, "new_token": new_token, "total_token": input_token + new_token, "stream": False}


def _session_id(req_dict: dict) -> dict:
    return OpenAIChatCompletionRequest.model_validate(req_dict)


def _parse_event(chunk: bytes) -> Optional[dict]:
    text = chunk.decode().strip()
    if not text.startswith("data: "):
        return None
    payload = text[len("data: "):]
    if payload == "[DONE]":
        return {"__done__": True}
    return json.loads(payload)


# ---------------------------------------------------------------------------
# flag off / no session -> unchanged behaviour
# ---------------------------------------------------------------------------

def test_no_session_usage_is_unchecked(monkeypatch: "pytest.MonkeyPatch") -> None:
    workers = _Workers(gen_result={"text": "hi", "metrics": _plain_metrics(120, 4)})

    async def _run():
        monkeypatch.setattr(R, "_sessions", SessionRegistry(None))
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        monkeypatch.setattr(R, "_workers", workers)
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        return await R.openai_chat_completions(req, _request(None))

    resp = asyncio.run(_run())
    # Exactly the pre-session response; no session_* leakage, per-request total.
    assert resp["usage"] == {"prompt_tokens": 120, "completion_tokens": 4, "total_tokens": 124}


def test_enabled_but_no_header_creates_no_session(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)
    workers = _Workers(gen_result={"text": "hi", "metrics": _plain_metrics(120, 4)})

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        monkeypatch.setattr(R, "_workers", workers)
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        # No header sent -> not a session for this request, despite the flag being on.
        return await R.openai_chat_completions(req, _request(None))

    asyncio.run(_run())
    assert reg.get("anything") is None


# ---------------------------------------------------------------------------
# session present: the reported "current context" is the session's value
# ---------------------------------------------------------------------------

def test_session_success_sets_session_id_and_reports_current_context(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    reg = SessionRegistry(HDR)
    # The worker measured an absolute (re)calibration of 5000 for this request.
    metrics = {"input_token": 5000, "new_token": 7, "total_token": 5007, "stream": False,
               "session_id": "sess-1", "session_context": 5000, "session_delta": None, "session_recalibrated": True}
    workers = _Workers(gen_result={"text": "ok", "metrics": metrics})

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        monkeypatch.setattr(R, "_workers", workers)
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        return await R.openai_chat_completions(req, _request("sess-1"))

    resp = asyncio.run(_run())
    # The number the client reads (goose: value before /context_window) is the session's.
    assert resp["usage"]["total_tokens"] == 5000
    assert resp["usage"]["prompt_tokens"] == 5000
    # completion_tokens keeps the REAL per-turn output (not 0), so goose's
    # accumulated output/cost accounting is unaffected by an active session.
    assert resp["usage"]["completion_tokens"] == 7
    # The session id was threaded onto the request the worker saw.
    assert workers.last_gen_config.session_id == "sess-1"
    assert reg.get("sess-1").current_context == 5000


def test_session_grows_across_requests(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        req = lambda: _session_id({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False})

        # Turn 1: a fresh session re-calibrates to the measured absolute (5000).
        R._workers = _Workers(gen_result={"text": "a", "metrics": {
            "input_token": 5000, "new_token": 3, "stream": False,
            "session_id": "sess-1", "session_context": 5000, "session_delta": None, "session_recalibrated": True}})
        r1 = await R.openai_chat_completions(req(), _request("sess-1"))
        # Turn 2: a continued turn grows the context by a delta (5000 -> 6800).
        R._workers = _Workers(gen_result={"text": "b", "metrics": {
            "input_token": 6800, "new_token": 5, "stream": False,
            "session_id": "sess-1", "session_context": 6800, "session_delta": 1800, "session_recalibrated": False}})
        r2 = await R.openai_chat_completions(req(), _request("sess-1"))
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1["usage"]["total_tokens"] == 5000
    assert r2["usage"]["total_tokens"] == 6800            # grew across requests
    assert reg.get("sess-1").current_context == 6800


# ---------------------------------------------------------------------------
# the fix itself: survive a worker restart, reporting the last-known value
# ---------------------------------------------------------------------------

def test_session_survives_nonstream_restart(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)  # flag on

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        # A prior healthy turn left the session at 4000.
        await reg.get_or_create("sess-1", MODEL)
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 4000, "session_delta": None})
        # Now the worker restarts while serving; it is still respawning (within budget).
        monkeypatch.setattr(R, "_workers", _Workers(raise_dead=True, will_respawn=True))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "trigger"}], "stream": False})
        return await R.openai_chat_completions(req, _request("sess-1"))

    resp = asyncio.run(_run())
    # The reported context is the last-known 4000, never 0; the session is untouched.
    assert resp["usage"]["total_tokens"] == 4000
    # A failed turn genuinely produced no output -> 0 is correct (and goose's
    # accumulated output is simply not advanced, not zeroed as a stale value).
    assert resp["usage"]["completion_tokens"] == 0
    assert reg.get("sess-1").current_context == 4000
    assert resp["choices"][0]["finish_reason"] == "error"  # the client knows it was a restart


def test_session_survives_stream_restart(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        await reg.get_or_create("sess-1", MODEL)
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 9000, "session_delta": None})
        # The worker restarts mid-stream within budget: one will_respawn error item.
        monkeypatch.setattr(R, "_workers", _Workers(stream_items=[{"error": "boom", "will_respawn": True}]))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "stream"}], "stream": True})
        resp = await R.openai_chat_completions(req, _request("sess-1"))
        events = [e for e in (await _drain(resp)) if e is not None and not e.get("__done__")]
        return events

    events = asyncio.run(_run())
    terminal = [e for e in events if "usage" in e]
    assert terminal, "the terminal event should carry a usage block"
    assert terminal[-1]["usage"]["total_tokens"] == 9000   # last-known, not 0
    assert reg.get("sess-1").current_context == 9000        # the failed turn did not clobber it


def test_session_rebases_on_resend(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        await reg.get_or_create("sess-1", MODEL)
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 9000, "session_delta": None})
        # goose re-sends the full conversation; a FRESH worker re-calibrates and
        # reports a new absolute (say the compacted 3000) -> the session re-bases.
        monkeypatch.setattr(R, "_workers", _Workers(gen_result={"text": "again", "metrics": {
            "input_token": 3000, "new_token": 2, "stream": False,
            "session_id": "sess-1", "session_context": 3000, "session_delta": None, "session_recalibrated": True}}))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "again"}], "stream": False})
        return await R.openai_chat_completions(req, _request("sess-1"))

    resp = asyncio.run(_run())
    assert resp["usage"]["total_tokens"] == 3000
    assert reg.get("sess-1").current_context == 3000


def test_session_rebases_on_compaction_shrink(monkeypatch: "pytest.MonkeyPatch") -> None:
    # The auto-compact loop, fixed. This is NOT a worker restart: goose compacts
    # away history, so the *same* live worker (still ready) re-measures a smaller
    # context and, with the fix, reports it recalibrated. The server re-bases the
    # session DOWN and hands the client that lower total_tokens -- so goose reads
    # itself back under its auto-compact threshold and stops compacting.
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        await reg.get_or_create("sess-1", MODEL)
        # Prior healthy turns left the session near its limit (the peak goose is
        # reacting to).
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 9000, "session_delta": None})
        reg.get("sess-1").apply({"session_recalibrated": False, "session_context": 9700, "session_delta": 700})
        assert reg.get("sess-1").current_context == 9700
        # goose compacts: the live worker re-measures a smaller context (3200) and
        # flags it recalibrated (a shrink). Same header -> same server session.
        monkeypatch.setattr(R, "_workers", _Workers(gen_result={"text": "done", "metrics": {
            "input_token": 3200, "new_token": 3, "stream": False,
            "session_id": "sess-1", "session_context": 3200, "session_delta": None, "session_recalibrated": True}}))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "after"}], "stream": False})
        return await R.openai_chat_completions(req, _request("sess-1"))

    resp = asyncio.run(_run())
    # The client reads the reset (compacted) context, not the pre-compaction peak.
    assert resp["usage"]["total_tokens"] == 3200
    assert reg.get("sess-1").current_context == 3200


def test_terminal_death_still_raises(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        await reg.get_or_create("sess-1", MODEL)
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 4000, "session_delta": None})
        # A terminal death (respawn budget spent, model unloaded) is NOT masked:
        # the client must still get an error, not a falsely healthy response.
        monkeypatch.setattr(R, "_workers", _Workers(raise_dead=True, will_respawn=False))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "trigger"}], "stream": False})
        return await R.openai_chat_completions(req, _request("sess-1"))

    with pytest.raises(HTTPException):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# streaming success folds the worker's per-request usage into the session
# ---------------------------------------------------------------------------

def test_stream_success_folds_session_usage(monkeypatch: "pytest.MonkeyPatch") -> None:
    reg = SessionRegistry(HDR)

    async def _run():
        monkeypatch.setattr(R, "_sessions", reg)
        monkeypatch.setattr(R, "_registry", _registry(m1={}))
        await reg.get_or_create("sess-1", MODEL)
        reg.get("sess-1").apply({"session_recalibrated": True, "session_context": 3000, "session_delta": None})
        # A happy streaming run whose final metrics continue the history (+1100).
        items = [{"metrics": {"input_token": 4100, "new_token": 2, "stream": True,
                              "session_id": "sess-1", "session_context": 4100,
                              "session_delta": 1100, "session_recalibrated": False}}, None]
        monkeypatch.setattr(R, "_workers", _Workers(stream_items=items))
        req = _session_id({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True})
        resp = await R.openai_chat_completions(req, _request("sess-1"))
        return [e for e in (await _drain(resp)) if e is not None and not e.get("__done__")]

    events = asyncio.run(_run())
    terminal = [e for e in events if "usage" in e]
    assert terminal[-1]["usage"]["total_tokens"] == 4100    # 3000 + 1100
    assert reg.get("sess-1").current_context == 4100


async def _drain(resp) -> list:
    out: list = []
    async for chunk in resp.body_iterator:
        parsed = _parse_event(chunk)
        if parsed is not None:
            out.append(parsed)
    return out
