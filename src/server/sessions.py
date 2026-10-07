"""Server-side session state: a proxy of a worker's ChatHistory usage.

Kept only when `openarc serve start --session-id-header/--sih` names the header
carrying a client session id (see SessionRegistry; a no-op when unset). It holds
just the "current context" number per session -- the messages stay in the
worker -- so a worker restart keeps reporting the last-known value (not 0), and
re-bases it when the client re-sends the full conversation and the worker
re-calibrates, *or* when the client compacts away history and the worker re-measures
a smaller context (a re-base that goes *down*, so goose reads itself back under its
auto-compact threshold instead of compacting forever).

Session is also the seam a future /v1/conversations endpoint would reuse
(get / delete / list); that endpoint is intentionally not built yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fastapi import Request


@dataclass
class ChatHistoryProxy:
    """Just the usage number that mirrors the worker's ChatHistory.

    `apply` is only ever called from a request the worker reported as
    successful; a crashed request reports nothing, so the value is left
    untouched (this is the whole point -- it survives a worker restart):
      * `session_recalibrated` True (or no delta) -> re-base to `session_context`.
        The worker raises a re-base in exactly two cases, and here it must go the
        right direction for both:
          - the process never saw the session (a restart/respawn, or a brand-new
            session): the last-known value is re-anchored to a *fresh* absolute;
          - the context shrank since the previous turn (goose compacted/trimmed the
            history, or re-templated): re-basing is what carries the *reset* value
            forward. Otherwise the number only ever climbs, the client keeps reading
            a pre-compaction peak over its threshold, and it compacts forever;
      * otherwise -> add `session_delta` (a continued history keeps climbing).
        (The worker now re-bases on a shrink, so a fold-in delta is always >= 0;
        the `max(0, ...)` guards below are a defensive floor for a stale input.)
    """

    current_context: int = 0
    turns: int = 0

    def apply(self, metrics: Optional[Dict[str, Any]]) -> int:
        if not metrics:
            return self.current_context
        if metrics.get("session_recalibrated") or metrics.get("session_delta") is None:
            # A re-base (restart, new session, or a compaction shrink): adopt the
            # absolute the worker measured -- this goes up or *down* accordingly, so
            # the number goose reads can follow a compaction down instead of peaking.
            ctx = metrics.get("session_context")
            if ctx is not None:
                self.current_context = max(0, int(ctx))
        else:
            # A continued (growing/flat) history folds its delta into the running
            # number; the worker keeps this delta >= 0, so the floor is defensive.
            self.current_context = max(0, self.current_context + int(metrics["session_delta"]))
        self.turns += 1
        return self.current_context


@dataclass
class Session:
    session_id: str
    model_name: str
    proxy: ChatHistoryProxy = field(default_factory=ChatHistoryProxy)
    created: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def current_context(self) -> int:
        return self.proxy.current_context

    def apply(self, metrics: Optional[Dict[str, Any]]) -> int:
        current = self.proxy.apply(metrics)
        self.updated_at = time.time()
        return current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "current_context": self.proxy.current_context,
            "turns": self.proxy.turns,
            "created": self.created,
            "updated_at": self.updated_at,
        }


class SessionRegistry:
    """In-memory sessions keyed by the header value. `enabled` is False (every
    method a no-op) unless a header is configured, so flag-off changes nothing."""

    def __init__(self, header: Optional[str] = None, max_sessions: int = 1024) -> None:
        self.reconfigure(header)
        self._max = max_sessions

    # (Re)set the source header; None/empty => sessions off, and clears any kept.
    def reconfigure(self, header: Optional[str]) -> None:
        self.header = header or None
        self._sessions: Dict[str, Session] = {}

    @property
    def enabled(self) -> bool:
        return self.header is not None

    def session_id_from(self, request: Optional[Request]) -> Optional[str]:
        if not self.enabled or request is None:
            return None
        value = request.headers.get(self.header)
        return value.strip() or None if value else None

    async def get_or_create(self, session_id: Optional[str], model_name: str) -> Optional[Session]:
        if not self.enabled or not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id, model_name=model_name)
            self._sessions[session_id] = session
            self._evict()
        else:
            session.model_name = model_name
            session.updated_at = time.time()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    # Light LRU-ish guard so an unbounded set of ids can't grow forever.
    def _evict(self) -> None:
        if len(self._sessions) <= self._max:
            return
        for s in sorted(self._sessions.values(), key=lambda x: x.created):
            self._sessions.pop(s.session_id, None)
            if len(self._sessions) <= self._max:
                return
