"""Worker-side per-session usage tracker.

The worker keeps one entry per session id and, for each request carrying one,
annotates that request's produced metrics with:
  * session_context      -- the absolute current context measured this turn
  * session_delta        -- how much that context grew since the previous turn in
                            THIS process; None when the server must re-base to
                            ``session_context`` instead of fold a delta in
  * session_recalibrated -- True when the server must re-base (set its context to
                            ``session_context`` rather than add a delta). That is
                            the case in exactly two situations:
                              1. this process has never seen the session -- a fresh
                                 worker after a crash/respawn, or a brand-new
                                 session, so only an absolute can re-anchor it;
                              2. the context shrank since the previous turn -- the
                                 *client* compacted (goose auto-compaction / the
                                 ``compact`` command) or otherwise trimmed the
                                 history. The running delta only ever climbs, so a
                                 shrink must re-base: the reset value the worker now
                                 reports is what the session has to carry forward, or
                                 the server keeps the pre-compaction peak, goose
                                 reads it back over its auto-compact threshold and
                                 the two keep compacting forever.

A server-side Session proxies it so the reported value survives a worker restart
and, just as importantly, tracks a compaction down. The conversation itself stays
in the (openvino_genai) ChatHistory in process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WorkerSession:
    context: int = 0
    turns: int = 0


class WorkerSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, WorkerSession] = {}

    def augment(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Add this session's usage to a produced metrics dict, in place. No-op
        (leaves the dict untouched) when it has no `input_token` -- only chat
        (LLM/VLM) requests carry one, so non-chat engines are unaffected."""
        if "input_token" not in metrics:
            return
        prev = self._sessions.get(session_id)
        context = int(metrics.get("input_token", 0))
        if prev is None or context < prev.context:
            # Re-base instead of folding a delta in. A session the process has never
            # seen (``prev is None`` -- a new session, or the first turn after a
            # crash/respawn) can only be re-anchored by an absolute. So too does a
            # SHRUNKEN context (``context < prev.context``): the client compacted away
            # history, so the worker's number is now *lower*. The additive approach
            # only grows, which would leave a stale, too-big peak in the session and
            # drive a goose auto-compact loop -- so in both cases we hand the server
            # the absolute and flag it recalibrated (delta None).
            delta: Optional[int] = None
            recalibrated = True
        else:
            # A continued history: fold the growth into the server's running number.
            # (Only reached when the context grew or held flat, so delta >= 0.)
            delta = context - prev.context
            recalibrated = False
        self._sessions[session_id] = WorkerSession(
            context=context,
            turns=(prev.turns if prev else 0) + 1,
        )
        metrics["session_id"] = session_id
        metrics["session_context"] = context
        metrics["session_delta"] = delta
        metrics["session_recalibrated"] = recalibrated
