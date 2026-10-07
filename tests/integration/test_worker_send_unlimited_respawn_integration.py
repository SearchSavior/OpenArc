"""A worker that dies on a broken pipe under the *unlimited* respawn policy
(``worker_max_respawns=0``) must log *short*, never a full "respawn budget is
exhausted" / "Exception in ASGI application" traceback.

That policy makes the supervisor's ``_will_respawn()`` unconditionally True
(``0 <= 0``: "the worker is always reloaded", so every death is a healing event),
yet ``WorkerSupervisor._send`` -- the one death site a request hits when the
pipe to a just-gone worker is already broken -- built its
``RemoteWorkerDeadError`` *without* ``will_respawn``, so it defaulted to a
terminal (loud) trace. This test reproduces it with a real ``WorkerSupervisor``
under ``worker_max_respawns=0`` and a broken pipe, then drives the *real*
``_commit_completed_packet`` and asserts the short healing line, no full traceback.

Run with: pytest -W ignore <this file>.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto
from src.engine.worker.supervisor import WorkerSupervisor
from src.server.worker_registry import WorkerPacket, _commit_completed_packet
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig

MODEL_NAME = "wedge-llm"
WORKER_LOGGER = "src.server.worker_registry"

# The one cause-only healing line vs the loud terminal / ASGI-escape markers that must NOT appear.
HEAL_MARKERS = (
    "restarting it within budget",
)
LOUD_MARKERS = (
    "respawn budget is exhausted",
    "Full traceback",
    "Unhandled exception",
    "Exception in ASGI application",
)


class _BrokenStdin:
    """A stdin pipe whose ``write`` fails -- exactly what a write to a worker
    that has died (before the supervisor's reader loop noticed the exit) does."""

    def write(self, data: bytes) -> None:
        raise BrokenPipeError("the inference worker process is gone; the pipe is broken")

    def close(self) -> None:
        pass

    async def drain(self) -> None:  # reached only on a successful write
        pass


class _FakeProc:
    """A stand-in for the worker subprocess handle: still-looking-alive
    (``returncode is None``) so ``_send``'s first guard passes and it *attempts*
    the write, which then hits the broken pipe and raises the ``cannot write to
    worker process`` ``RemoteWorkerDeadError`` we are testing."""

    returncode: Optional[int] = None

    def __init__(self) -> None:
        self.pid = 4242
        self.stdin = _BrokenStdin()
        self.stdout = None
        self.stderr = None

    async def wait(self) -> int:
        return self.returncode or -1


class _RecordingRegistry:
    """A minimal ModelRegistry stand-in: _commit_completed_packet only touches a
    registry via ``register_unload(model_name)`` (in the *terminal* branch it
    does NOT take -- a dead worker is left to the supervisor, per the registry's
    own comment), so recording those calls lets us assert the model was neither
    unloaded nor quarantined for a healing in-budget death."""

    def __init__(self) -> None:
        self.unloaded: List[str] = []

    async def register_unload(self, model_name: str, *args: Any, **kwargs: Any) -> bool:
        self.unloaded.append(model_name)
        return True


def _unlimited_supervisor_with_broken_pipe(model_name: str) -> WorkerSupervisor:
    """A real ``WorkerSupervisor`` under the unlimited respawncap, whose proc's
    stdin pipe is already broken and whose state is READY (so ``begin_run`` would
    have proceeded to ``_send`` -- the in-flight-request path the operator hit)."""
    sup = WorkerSupervisor(model_name, max_respawns=0)
    sup._state = sup.STATE_READY
    sup._proc = _FakeProc()
    return sup


def _server_logs(caplog: Any) -> list:
    return [r for r in caplog.records if r.name == WORKER_LOGGER]


def test_worker_send_to_dead_worker_under_unlimited_budget_logs_quietly(
    monkeypatch: "pytest.MonkeyPatch", caplog: Any,
) -> None:
    caplog.set_level(logging.DEBUG)

    # The policy under test: worker_max_respawns == 0 = "no limit, always
    # reload", for which the supervisor's _will_respawn() is unconditionally
    # True. Proven at the source so the test's premise is self-documenting.
    async def _run() -> dict:
        sup = _unlimited_supervisor_with_broken_pipe(MODEL_NAME)
        assert sup._max_respawns == 0
        assert sup._will_respawn() is True  # unlimited -> every death is a heal

        # The dead worker's broken pipe: _send is the real, now-fixed death site.
        err: Optional[BaseException] = None
        try:
            await sup._send(proto.encode("request", req_id="r", request_id=None, gen_config="{}"))
        except proto.RemoteWorkerDeadError as e:
            err = e

        # The fix's direct effect: _send carries the heal flag it now consults.
        assert isinstance(err, proto.RemoteWorkerDeadError), (
            f"_send on a broken pipe should raise RemoteWorkerDeadError, not {err!r}"
        )
        assert "cannot write to worker process" in str(err), str(err)
        assert err.will_respawn is True, (
            "under the unlimited (worker_max_respawns=0) policy a dead worker is "
            "being reloaded within budget, so _send must carry will_respawn=True; "
            "a False here is the regression that made every such death fall "
            "through to a full 'respawn budget is exhausted' trace"
        )

        # And the observable the operator actually sees: feed that exact error
        # through the registry's real completion/decision path and assert the log
        # comes out as the single short healing line -- not the loud terminal one.
        packet = WorkerPacket(
            request_id="r",
            id_model=MODEL_NAME,
            gen_config=OVGenAI_GenConfig(),
            stream_queue=None,
            result_future=None,  # no result to complete for this probe
        )
        packet.error = err
        reg = _RecordingRegistry()
        should_exit = _commit_completed_packet(packet, packet, MODEL_NAME, reg)
        await asyncio.sleep(0)  # let any (should-be-absent) register_unload task run
        return {
            "should_exit": should_exit,
            "unloaded": reg.unloaded,
        }

    result = asyncio.run(_run())

    server_lines = _server_logs(caplog)

    # The healing line is present ...
    join = "\n".join(r.getMessage() for r in server_lines)
    assert any(any(m in r.getMessage() for m in HEAL_MARKERS) for r in server_lines), (
        f"expected the short healing line for a within-budget respawn; got: {server_lines}"
    )
    # ... and no loud / full-trace marker anywhere in the server log.
    assert not any(m in join for m in LOUD_MARKERS), (
        f"a healing (within-budget) respawn must not log a full trace; found a loud "
        f"marker: {server_lines}"
    )
    assert not any(r.exc_info for r in server_lines), (
        f"a healing (within-budget) respawn must not attach a traceback; got: "
        f"{[r.getMessage() for r in server_lines if r.exc_info]}"
    )
    # ... a dead worker is left to the supervisor, so the registry neither
    # unloads it nor signals the worker should exit.
    assert result["should_exit"] is False
    assert result["unloaded"] == []
    print(f"[{MODEL_NAME}] heal log: {server_lines[0].getMessage()}")

