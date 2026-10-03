"""Unit tests for the worker-side per-session usage tracker (worker/session.py).

The worker keeps one entry per session id and, for each request carrying a
session id, annotates that request's produced metrics with:
  * session_context      -- the absolute current context measured this turn
  * session_delta        -- growth since the previous turn in THIS process; None
                            when the server re-bases instead (a first/respawn
                            turn, or a compaction shrink)
  * session_recalibrated -- True when the server must re-base to session_context
                            rather than add a delta: a fresh worker (a crash/respawn,
                            or a brand-new session), OR the context having shunk since
                            the previous turn (the client compacted). Re-basing on a
                            shrink is what lets the server-side number follow a
                            compaction down instead of peaking and looping.
"""

from __future__ import annotations

from src.engine.worker.session import WorkerSessionManager


def _metrics(input_token: int, **extra) -> dict:
    m = {"input_token": input_token, "new_token": 3, "stream": False}
    m.update(extra)
    return m


def test_first_turn_recalibrates() -> None:
    m = WorkerSessionManager()
    metrics = _metrics(5000)
    m.augment("sess-1", metrics)
    assert metrics["session_id"] == "sess-1"
    assert metrics["session_context"] == 5000
    assert metrics["session_delta"] is None        # no prior turn
    assert metrics["session_recalibrated"] is True


def test_continued_turn_reports_delta() -> None:
    m = WorkerSessionManager()
    m.augment("sess-1", _metrics(5000))
    metrics = _metrics(6800)
    m.augment("sess-1", metrics)
    assert metrics["session_context"] == 6800
    assert metrics["session_delta"] == 1800        # 6800 - 5000
    assert metrics["session_recalibrated"] is False


def test_fresh_process_resees_session_as_recalibrated() -> None:
    # The key property: a worker process has a fresh, empty tracker, so when it
    # sees a session it already knows about (goose re-sent the conversation after
    # a crash) it reports recalibrated=True -> the server re-bases, not adds.
    m = WorkerSessionManager()
    metrics = _metrics(9000)
    m.augment("sess-1", metrics)
    assert metrics["session_recalibrated"] is True
    assert metrics["session_delta"] is None


def test_independent_sessions() -> None:
    m = WorkerSessionManager()
    a = _metrics(1000)
    b = _metrics(2000)
    m.augment("sa", a)
    m.augment("sb", b)
    # sa first-seen -> recalibrate; sb first-seen -> recalibrate (independent).
    assert a["session_recalibrated"] is True
    assert b["session_recalibrated"] is True
    a2 = _metrics(1500)
    b2 = _metrics(3500)
    m.augment("sa", a2)
    m.augment("sb", b2)
    assert a2["session_delta"] == 500
    assert b2["session_delta"] == 1500
    assert a2["session_recalibrated"] is False
    assert b2["session_recalibrated"] is False


def test_no_input_token_is_a_noop() -> None:
    # Engines that never report a context (embeddings, rerank, ...) must be
    # untouched: their metrics carry no input_token, so no session keys are added
    # and nothing is tracked. (Sessions only ever arrive with LLM/VLM chat.)
    m = WorkerSessionManager()
    metrics = {"prompt_tokens": 10, "stream": False}
    m.augment("sess-1", metrics)
    assert "session_context" not in metrics
    assert "session_delta" not in metrics
    assert "session_recalibrated" not in metrics
    assert m._sessions == {}


def test_shrink_rebases() -> None:
    # The compaction case: goose compacts (or trims) away history, so the worker
    # measures a SMALLER context this turn than the previous one. The running delta
    # only climbs, so a shrink must re-base (recalibrated True, delta None) -- the
    # server adopts the new (lower) absolute, else it keeps the pre-compaction peak
    # and goose keeps compacting forever.
    m = WorkerSessionManager()
    m.augment("sess-1", _metrics(5000))          # turn 1
    metrics = _metrics(1000)                      # turn 2: context shrank (compaction)
    m.augment("sess-1", metrics)
    assert metrics["session_context"] == 1000
    assert metrics["session_delta"] is None      # re-base, not a (clamped) delta
    assert metrics["session_recalibrated"] is True


def test_flat_turn_is_zero_growth() -> None:
    # Context unchanged between turns -> a zero delta, still a continued (not
    # re-based) turn; the server adds 0 and the number is simply unchanged.
    m = WorkerSessionManager()
    m.augment("sess-1", _metrics(5000))
    metrics = _metrics(5000)
    m.augment("sess-1", metrics)
    assert metrics["session_delta"] == 0
    assert metrics["session_recalibrated"] is False
