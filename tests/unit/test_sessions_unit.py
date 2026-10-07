"""Unit tests for the server-side session proxy (src/server/sessions.py).

The proxy mirrors a worker's ChatHistory usage ("current context") so that a
worker restart keeps reporting the last-known value instead of 0. No worker or
event loop is needed: the proxy is a plain object feeding off the worker's
per-request metrics keys (session_context / session_delta / session_recalibrated).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import Request

from src.server.sessions import ChatHistoryProxy, Session, SessionRegistry

HDR = "X-Session-Id"


def _request(hdr_value: Optional[str] = None) -> Request:
    """A minimal ASGI request whose only relevant part is its headers."""
    headers = [] if hdr_value is None else [(HDR.lower().encode(), hdr_value.encode())]

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
        receive=_receive,
    )


# ---------------------------------------------------------------------------
# disabled (flag off): a strict no-op
# ---------------------------------------------------------------------------

def test_disabled_registry_is_a_noop() -> None:
    reg = SessionRegistry(None)
    assert reg.enabled is False
    # No header in effect -> nothing is ever read, even when the client sends one.
    assert reg.session_id_from(_request("sess-1")) is None
    assert reg.session_id_from(_request()) is None
    # get_or_create must not create anything while disabled.
    assert asyncio.run(reg.get_or_create("sess-1", "m1")) is None
    assert reg.get("sess-1") is None


def test_empty_header_string_means_disabled() -> None:
    # `serve start` exports "" when --sih is omitted; that must read as disabled.
    reg = SessionRegistry("")
    assert reg.enabled is False
    assert asyncio.run(reg.get_or_create("sess-1", "m1")) is None


# ---------------------------------------------------------------------------
# enabled (flag on): read the header, keep one session per id
# ---------------------------------------------------------------------------

def test_enabled_reads_header_from_request() -> None:
    reg = SessionRegistry(HDR)
    assert reg.enabled is True
    assert reg.session_id_from(_request("sess-7")) == "sess-7"
    assert reg.session_id_from(_request()) is None          # no header -> None
    assert reg.session_id_from(_request("   ")) is None      # blank -> None
    assert reg.session_id_from(None) is None                 # no request -> None


def test_get_or_create_persists_same_object() -> None:
    reg = SessionRegistry(HDR)
    loop = asyncio.new_event_loop()
    try:
        a = loop.run_until_complete(reg.get_or_create("sess-1", "m1"))
        b = loop.run_until_complete(reg.get_or_create("sess-1", "m1"))
        c = loop.run_until_complete(reg.get_or_create("sess-2", "m1"))
    finally:
        loop.close()
    assert a is b                                            # same session reused
    assert a is not c
    assert isinstance(a, Session)
    assert a.session_id == "sess-1"


# ---------------------------------------------------------------------------
# the proxy number: the heart of the feature
# ---------------------------------------------------------------------------

def test_proxy_recalibrate_then_increment_then_retain() -> None:
    p = ChatHistoryProxy()
    # First turn a (re)calibrated session sets it to the measured absolute.
    assert p.apply({"session_recalibrated": True, "session_context": 5000, "session_delta": None}) == 5000
    # A continued turn adds the delta (the value grows across requests).
    assert p.apply({"session_recalibrated": False, "session_context": 5700, "session_delta": 700}) == 5700
    assert p.apply({"session_recalibrated": False, "session_context": 6800, "session_delta": 1100}) == 6800
    # A failed request reports nothing -> apply is not called -> value is retained.
    assert p.current_context == 6800


def test_death_then_resend_rebases() -> None:
    # The scenario under test: build up a healthy value, the worker dies (nothing
    # updates the proxy), then goose re-sends the full conversation and the worker
    # re-calibrates -> the session is re-based to that fresh measurement.
    p = ChatHistoryProxy()
    p.apply({"session_recalibrated": True, "session_context": 6000, "session_delta": None})
    p.apply({"session_recalibrated": False, "session_context": 7200, "session_delta": 1200})
    assert p.current_context == 7200
    # Worker went away: the session simply holds 7200.
    # goose re-sends; a fresh worker reports a re-measured absolute (maybe goat
    # compacted, so a *different* number than 7200).
    assert p.apply({"session_recalibrated": True, "session_context": 4100, "session_delta": None}) == 4100
    assert p.current_context == 4100


def test_growth_branch_floor() -> None:
    # The worker now re-bases on a shrink (recalibrated True) instead of sending a
    # negative delta, so this path is purely defensive: a fold-in delta is always
    # >= 0, but the proxy must still never let a stale/negative one drive the
    # reported number below 0.
    p = ChatHistoryProxy()
    p.apply({"session_recalibrated": False, "session_context": 1000, "session_delta": -100000})
    assert p.apply({"session_recalibrated": False, "session_context": 200, "session_delta": -100000}) == 0
    assert p.current_context >= 0


def test_shrink_rebases_in_proxy() -> None:
    # The compaction case on the server side: a live session (no worker restart)
    # whose worker reports a SHRUNKEN context because goose compacted away history.
    # The worker now flags that turn recalibrated, so the proxy re-bases DOWN to
    # the reset value instead of holding the pre-compaction peak -- otherwise goose
    # keeps reading the peak over its auto-compact threshold and compacts forever.
    p = ChatHistoryProxy()
    p.apply({"session_recalibrated": True, "session_context": 9000, "session_delta": None})
    p.apply({"session_recalibrated": False, "session_context": 10400, "session_delta": 1400})
    assert p.current_context == 10400
    # goose compacts: the worker re-sends a smaller conversation (smaller context)
    # and, with the fix, flags it recalibrated so the proxy drops to the reset value.
    assert p.apply({"session_recalibrated": True, "session_context": 3200, "session_delta": None}) == 3200
    assert p.current_context == 3200


# ---------------------------------------------------------------------------
# registry plumbing for a future /v1/conversations

def test_get_delete_and_list() -> None:
    reg = SessionRegistry(HDR)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(reg.get_or_create("sess-1", "llm"))
        loop.run_until_complete(reg.get_or_create("sess-2", "llm"))
        assert {e["session_id"] for e in reg.list()} == {"sess-1", "sess-2"}
        reg.delete("sess-1")
        assert reg.get("sess-1") is None
        assert reg.get("sess-2") is not None
    finally:
        loop.close()


def test_eviction_drops_oldest_when_over_cap() -> None:
    reg = SessionRegistry(HDR, max_sessions=2)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(reg.get_or_create("s1", "m"))
        loop.run_until_complete(reg.get_or_create("s2", "m"))
        loop.run_until_complete(reg.get_or_create("s3", "m"))
    finally:
        loop.close()
    assert len(reg._sessions) == 2
    assert reg.get("s1") is None          # oldest evicted
    assert reg.get("s2") is not None
    assert reg.get("s3") is not None


def test_reconfigure_toggles_and_clears() -> None:
    reg = SessionRegistry(HDR)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(reg.get_or_create("sess-1", "m"))
        assert reg.enabled is True
        reg.reconfigure(None)
        assert reg.enabled is False
        assert reg.get("sess-1") is None    # sessions cleared on reconfigure
        reg.reconfigure(HDR)
        assert reg.enabled is True
    finally:
        loop.close()
