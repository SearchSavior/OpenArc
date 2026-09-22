"""
WorkerSupervisor: owns one inference worker subprocess for one model.

Responsibilities:
  * spawn the child (``src.engine.worker.worker_process``) with the same
    interpreter, sys.path and working directory the parent runs with,
  * speak the line protocol on the child's stdin/stdout,
  * forward the child's stderr to the openarc log (OpenVINO/OpenCL output),
  * watch the process: PING watchdog, death detection, and -- when the child
    dies while serving -- respawn a fresh process and re-run the same load,
    within a per-load budget (a fresh process gets a fresh, unpoisoned
    ov::Core). Once the budget is exhausted the ``on_dead`` callback fires,
    which the facade wires to a model unload.

State machine:
    NOT_STARTED -> STARTING -> LOADING -> READY -> (RESTARTING -> READY)
                                            \\-> DEAD  (budget exhausted / load episode over)
    any -> CLOSED (explicit unload; terminal)

Load-episode rule: if the process dies while the initial load is still in
flight, no respawn happens in the background -- the registry's load call gets
the error and decides; a background respawn would leave a live process with
no registry record behind.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union

from src.engine.worker import protocol as proto
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

# Sentinel the supervisor puts on a request queue to tell the client-side
# generator: "the stream has ended; now await the result future".
EOF = object()


class _PendingRequest:
    """Supervisor side of one GENERATE: item queue + result future."""

    __slots__ = ("req_id", "queue", "result")

    def __init__(self, req_id: str) -> None:
        self.req_id = req_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.result: Optional[asyncio.Future] = None


class WorkerSupervisor:
    """Spawns, talks to, and watches one inference worker subprocess."""

    STATE_NOT_STARTED = "not_started"
    STATE_STARTING = "starting"
    STATE_LOADING = "loading"
    STATE_READY = "ready"
    STATE_RESTARTING = "restarting"
    STATE_DEAD = "dead"
    STATE_CLOSED = "closed"

    def __init__(
        self,
        model_name: str,
        *,
        max_respawns: int = 2,
        load_timeout: Optional[float] = None,
        unload_timeout: float = 10.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 5.0,
        restart_backoff: float = 0.5,
        on_dead: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._model_name = model_name
        self._max_respawns = max_respawns
        self._load_timeout = load_timeout
        self._unload_timeout = unload_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._restart_backoff = restart_backoff
        self._on_dead = on_dead

        self._state = self.STATE_NOT_STARTED
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._respawn_task: Optional[asyncio.Task] = None
        self._load_future: Optional[asyncio.Future] = None
        self._ping_future: Optional[asyncio.Future] = None
        self._cancel: Optional[Tuple[str, asyncio.Future]] = None
        self._active: Dict[str, _PendingRequest] = {}
        self._fatal_error: Optional[str] = None
        self._death_handled = False
        self._closed = False
        self._respawns = 0
        self._load_config: Optional[ModelLoadConfig] = None

    # -- introspection --------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def respawns(self) -> int:
        return self._respawns

    def status(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "pid": self.pid,
            "respawns": self._respawns,
        }

    # -- lifecycle -------------------------------------------------------------
    async def start(self, load_config: ModelLoadConfig, *, load_timeout: Optional[float] = None) -> None:
        """Spawn the worker process and load the model inside it.

        Raises RemoteWorkerLoadError / RemoteWorkerDeadError when the load
        does not complete; the child process is always cleaned up first.
        """
        if self._state not in (self.STATE_NOT_STARTED, self.STATE_DEAD):
            raise RuntimeError(f"supervisor is already running (state={self._state})")
        self._load_config = load_config
        self._respawns = 0
        self._closed = False
        timeout = self._load_timeout if load_timeout is None else load_timeout
        try:
            await self._spawn_and_load(timeout)
        except asyncio.CancelledError:
            self._abort_child()
            raise
        except Exception:
            self._abort_child()
            raise

    def _abort_child(self) -> None:
        """Best-effort cleanup after a failed load episode."""
        self._closed = True
        self._set_state(self.STATE_DEAD)
        self._stop_ping()
        self._terminate_process()

    # -- spawning ---------------------------------------------------------------
    def _build_command(self) -> list:
        return [
            sys.executable,
            "-c",
            "import sys; from src.engine.worker.worker_process import main; sys.exit(main())",
        ]

    def _build_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        # The child must import exactly the code the parent is running,
        # whatever the install mode (editable, wheel, plain cwd): hand over
        # the parent's sys.path and keep the working directory.
        paths = [p for p in sys.path if p]
        cwd = os.getcwd()
        if cwd not in paths:
            paths.insert(0, cwd)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    async def _spawn(self) -> None:
        command = self._build_command()
        logger.info(f"[{self._model_name}] spawning inference worker: {' '.join(command)}")
        self._proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env=self._build_env(),
        )
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"ovworker-read-{self._model_name}"
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_pump(), name=f"ovworker-stderr-{self._model_name}"
        )

    async def _spawn_and_load(self, timeout: Optional[float]) -> None:
        if self._closed:
            raise proto.RemoteWorkerDeadError("supervisor is closed")
        self._set_state(self.STATE_STARTING)
        self._fatal_error = None
        self._death_handled = False
        await self._spawn()
        self._start_ping()
        self._load_future = asyncio.get_running_loop().create_future()
        self._set_state(self.STATE_LOADING)
        await self._send(
            proto.encode(
                proto.OP_LOAD, req_id="load", config=self._load_config.model_dump_json()
            )
        )
        try:
            await asyncio.wait_for(self._load_future, timeout)
        except asyncio.TimeoutError as e:
            raise proto.RemoteWorkerLoadError(
                f"worker did not finish loading within {timeout:.0f}s"
            ) from e
        self._set_state(self.STATE_READY)
        logger.info(f"[{self._model_name}] inference worker ready (pid={self.pid})")

    # -- sending -----------------------------------------------------------------
    async def _send(self, data: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise proto.RemoteWorkerDeadError("worker process is not running")
        try:
            proc.stdin.write(data)
            await proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, ValueError, OSError) as e:
            raise proto.RemoteWorkerDeadError(f"cannot write to worker process: {e}") from e

    # -- request API ----------------------------------------------------------------
    async def begin_generate(
        self, gen_config_json: str, request_id: Optional[str]
    ) -> Tuple[asyncio.Queue, asyncio.Future]:
        """Send GENERATE and return (item queue, result future).

        The queue yields the items the model's generate_type produced; it
        always ends with the EOF sentinel, after which the result future
        resolves (None) or raises.
        """
        if self._state != self.STATE_READY:
            raise proto.RemoteWorkerDeadError(
                f"inference worker not ready (state={self._state})"
            )
        loop = asyncio.get_running_loop()
        req = _PendingRequest(uuid.uuid4().hex)
        req.result = loop.create_future()
        self._active[req.req_id] = req
        await self._send(
            proto.encode(
                proto.OP_GENERATE,
                req_id=req.req_id,
                request_id=request_id,
                gen_config=gen_config_json,
            )
        )
        return req.queue, req.result

    async def request_cancel(self, request_id: str) -> bool:
        """Ask the worker to cancel the generation tracked by request_id."""
        if self._state != self.STATE_READY:
            return False
        loop = asyncio.get_running_loop()
        req_id = uuid.uuid4().hex
        fut: asyncio.Future = loop.create_future()
        self._cancel = (req_id, fut)
        try:
            await self._send(proto.encode(proto.OP_CANCEL, req_id=req_id, request_id=request_id))
            return bool(await asyncio.wait_for(fut, 5.0))
        except asyncio.TimeoutError:
            logger.warning(f"[{self._model_name}] no CANCEL ack from worker")
            return False
        except proto.RemoteWorkerDeadError:
            return False
        finally:
            if self._cancel is not None and self._cancel[0] == req_id:
                self._cancel = None

    # -- unload ---------------------------------------------------------------------
    async def unload(self) -> None:
        """Ask the worker to exit and wait for it (terminate/kill as backup).

        Idempotent: always converges to the CLOSED state and reaps the pipe
        tasks, whether called after a healthy episode, a failed load, or
        repeated calls.
        """
        self._closed = True
        self._stop_ping()
        self._set_state(self.STATE_CLOSED)
        if self._respawn_task is not None and not self._respawn_task.done():
            self._respawn_task.cancel()
            try:
                await self._respawn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._respawn_task = None
        if self._load_future is not None and not self._load_future.done():
            self._load_future.set_exception(proto.RemoteWorkerDeadError("worker unloaded"))
        self._fail_all_active(proto.RemoteWorkerDeadError("worker unloaded"))
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                await self._send(proto.encode(proto.OP_UNLOAD, req_id="unload"))
            except proto.RemoteWorkerDeadError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), self._unload_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{self._model_name}] worker did not exit after UNLOAD; terminating"
                )
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self._model_name}] worker did not exit after SIGTERM; killing"
                    )
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
        await self._await_pipe_tasks()
        self._close_process_streams()
        logger.info(f"[{self._model_name}] inference worker process stopped")

    async def _await_pipe_tasks(self) -> None:
        for task in (self._reader_task, self._stderr_task):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._reader_task = None
        self._stderr_task = None

    def _close_process_streams(self) -> None:
        """Release the subprocess transport so its pipes are closed.

        StreamReader (stdout/stderr) has no close() of its own; closing the
        transport closes every pipe, delivers EOF to any pending reader, and
        prevents the "Event loop is closed" warning when the transport is
        otherwise only torn down by GC after the loop has shut down.
        Idempotent: safe to call repeatedly.
        """
        proc = self._proc
        if proc is None:
            return
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except (RuntimeError, ValueError):
                pass

    # -- protocol session ---------------------------------------------------------------
    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = proto.decode_response(line)
                except proto.ProtocolError as e:
                    logger.warning(f"[{self._model_name}] worker protocol error: {e}")
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[{self._model_name}] worker reader loop crashed")
        await proc.wait()
        self._close_process_streams()
        self._on_process_exited(proc.returncode)

    async def _stderr_pump(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.info(f"[{self._model_name} pid={proc.pid}] {text}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[{self._model_name}] worker stderr pump failed")

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        mtype = msg["type"]
        if mtype == proto.MSG_READY:
            if self._ping_future is not None and not self._ping_future.done():
                self._ping_future.set_result(True)
        elif mtype == proto.MSG_LOAD_OK:
            if self._load_future is not None and not self._load_future.done():
                self._load_future.set_result(None)
        elif mtype == proto.MSG_LOAD_ERROR:
            err = msg.get("error") or {}
            self._fail_load(
                proto.RemoteWorkerLoadError(
                    err.get("message") or "worker reported a load failure",
                    original_type=err.get("type"),
                )
            )
        elif mtype == proto.MSG_ITEM:
            req = self._active.get(msg.get("req_id"))
            if req is not None and not (req.result is not None and req.result.done()):
                req.queue.put_nowait(msg.get("item"))
        elif mtype == proto.MSG_DONE:
            self._finish_request(msg.get("req_id"), None)
        elif mtype == proto.MSG_ERROR:
            err = msg.get("error") or {}
            self._finish_request(
                msg.get("req_id"),
                proto.RemoteWorkerError(
                    err.get("message") or "inference failed in worker",
                    original_type=err.get("type"),
                ),
            )
        elif mtype == proto.MSG_FATAL:
            err = msg.get("error") or {}
            self._fatal_error = err.get("message") or "worker reported a fatal error"
            # The worker is taking itself down on purpose (wedged device): for
            # every in-flight request the worker IS dead, so the failures are
            # RemoteWorkerDeadErrors -- the supervisor owns recovery (respawn
            # within budget, or an unload once it is exhausted), and the
            # registry must not double-act by unloading on the request error.
            exc = proto.RemoteWorkerDeadError(self._fatal_error, original_type=err.get("type"))
            self._fail_load(exc)
            self._fail_all_active(exc)
        elif mtype == proto.MSG_BYE:
            pass  # the process exits right after; the reader loop handles it
        elif mtype == proto.MSG_CANCEL_ACK:
            if (
                self._cancel is not None
                and self._cancel[0] == msg.get("req_id")
                and not self._cancel[1].done()
            ):
                self._cancel[1].set_result(bool(msg.get("ok")))

    def _finish_request(self, req_id: Optional[str], error: Optional[BaseException]) -> None:
        req = self._active.pop(req_id, None)
        if req is None:
            return
        req.queue.put_nowait(EOF)
        if req.result is not None and not req.result.done():
            if error is None:
                req.result.set_result(None)
            else:
                req.result.set_exception(error)

    def _fail_load(self, exc: BaseException) -> None:
        if self._load_future is not None and not self._load_future.done():
            self._load_future.set_exception(exc)

    def _fail_all_active(self, exc: BaseException) -> None:
        for req_id in list(self._active):
            self._finish_request(req_id, exc)

    # -- death & respawn ------------------------------------------------------------------
    def _on_process_exited(self, code: Optional[int]) -> None:
        if self._death_handled:
            return
        self._death_handled = True
        self._stop_ping()
        detail = f" (code={code})"
        suffix = f": {self._fatal_error}" if self._fatal_error else ""
        if self._closed:
            # Clean unload path: nothing to recover, nothing to respawn.
            self._fail_load(proto.RemoteWorkerDeadError(f"worker process exited{detail}"))
            return
        logger.error(
            f"[{self._model_name}] inference worker exited unexpectedly{detail}{suffix}"
        )
        death_error = proto.RemoteWorkerDeadError(
            f"inference worker process exited unexpectedly{detail}{suffix}"
        )
        was_loading = self._load_future is not None and not self._load_future.done()
        self._fail_load(death_error)
        self._fail_all_active(death_error)
        if was_loading or self._state == self.STATE_LOADING:
            # Load episode over: the registry's load call gets the error and
            # decides. A background respawn would leave a live process with
            # no registry record behind.
            self._set_state(self.STATE_DEAD)
            return
        if self._load_config is None:
            self._set_state(self.STATE_DEAD)
            return
        if self._respawns < self._max_respawns:
            self._respawns += 1
            self._set_state(self.STATE_RESTARTING)
            logger.warning(
                f"[{self._model_name}] respawning inference worker "
                f"({self._respawns}/{self._max_respawns})"
            )
            self._respawn_task = asyncio.create_task(
                self._respawn(), name=f"ovworker-respawn-{self._model_name}"
            )
        else:
            self._set_state(self.STATE_DEAD)
            logger.error(
                f"[{self._model_name}] respawn budget exhausted; worker is dead"
            )
            if self._on_dead is not None:
                asyncio.create_task(self._fire_on_dead())

    async def _respawn(self) -> None:
        try:
            await asyncio.sleep(self._restart_backoff)
            await self._spawn_and_load(self._load_timeout)
            logger.info(
                f"[{self._model_name}] inference worker respawned (pid={self.pid})"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._closed:
                return
            logger.error(f"[{self._model_name}] worker respawn failed: {e}")
            self._set_state(self.STATE_DEAD)
            if self._on_dead is not None:
                await self._fire_on_dead()

    async def _fire_on_dead(self) -> None:
        try:
            await self._on_dead()
        except Exception:
            logger.exception(f"[{self._model_name}] on_dead callback failed")

    # -- watchdog ---------------------------------------------------------------------------
    def _start_ping(self) -> None:
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(
                self._ping_loop(), name=f"ovworker-ping-{self._model_name}"
            )

    def _stop_ping(self) -> None:
        if self._ping_task is not None:
            self._ping_task.cancel()
            self._ping_task = None

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            if self._closed or self._state not in (self.STATE_LOADING, self.STATE_READY):
                return
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._ping_future = fut
            try:
                await self._send(proto.encode(proto.OP_PING, req_id="ping"))
                await asyncio.wait_for(fut, self._ping_timeout)
            except asyncio.CancelledError:
                return
            except proto.RemoteWorkerDeadError:
                return  # the write failed; the reader loop handles the death
            except asyncio.TimeoutError:
                logger.error(
                    f"[{self._model_name}] worker unresponsive to PING for "
                    f"{self._ping_timeout:.0f}s; killing the process"
                )
                self._terminate_process()
                return

    def _terminate_process(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def _set_state(self, state: str) -> None:
        if state != self._state:
            logger.debug(f"[{self._model_name}] worker state {self._state} -> {state}")
            self._state = state
