"""
WorkerSupervisor: owns one inference worker subprocess for one model.

Responsibilities:
  * spawn the child (``src.engine.worker.worker_process``) with the same
    interpreter, sys.path and working directory the parent runs with,
  * speak the line protocol on the child's stdin/stdout,
  * hand each worker its own log file over the environment, so the worker itself
    writes its logging and native (C-level) stderr there -- nothing of its output
    reaches the openarc log,
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
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Tuple, Union

from src.engine.worker import platform_support
from src.engine.worker import protocol as proto
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

# Sentinel the supervisor puts on a request queue to tell the client-side
# generator: "the stream has ended; now await the result future".
EOF = object()


def _sanitize_for_filename(name: str) -> str:
    """Make a model name safe as a file-name fragment: the worker's log-file
    name is derived from the model name verbatim (per the
    "<base>-worker-<model>.log" rule), but a path separator in it would let the
    file escape the log directory, so reduce "/", "\\", and NUL to "-".
    """
    for separator in ("/", "\\", "\x00", os.sep):
        name = name.replace(separator, "-")
    return name


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
        platform: Optional[platform_support.Platform] = None,
    ) -> None:
        self._model_name = model_name
        # The host this supervisor runs on, chosen once (the spawn flags, the
        # PYTHONPATH separator, and the terminate strategy all key off it).
        # Defaults to the real host; a test may inject the *other* platform to
        # drive its pure decisions (see platform_support) without a real host.
        self._platform = platform if platform is not None else platform_support.current_platform()
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
        self._line_limit = proto.PROTOCOL_LINE_LIMIT

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
            "max_respawns": self._max_respawns,
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
        self._line_limit = (
            load_config.worker_line_limit
            if load_config.worker_line_limit is not None
            else proto.PROTOCOL_LINE_LIMIT
        )
        # The respawn budget is per load episode; a per-model override
        # (worker_max_respawns, see --worker-max-respawns) wins over the
        # constructor default (2), exactly like the line limit. 0 or a negative
        # number means "no limit" (the worker is always reloaded).
        self._max_respawns = (
            load_config.worker_max_respawns
            if load_config.worker_max_respawns is not None
            else self._max_respawns
        )
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

    # Child entry module (the one that provides main()). The plain-OpenVINO
    # worker (src.engine.worker.plain) subclasses and overrides this.
    WORKER_ENTRY = "src.engine.worker.worker_process"

    # -- spawning ---------------------------------------------------------------
    def _build_command(self) -> list:
        return [
            sys.executable,
            "-c",
            f"import sys; from {self.WORKER_ENTRY} import main; sys.exit(main())",
        ]

    def _worker_logfile_env(self) -> str:
        """The env key the supervisor hands the worker its log file in:
        OPENARC_WORKER_LOGFILE_<model>, the dash rewritten to an underscore (an
        env-var name rule) so one key names one worker. The worker rebuilds the
        SAME key from OPENARC_WORKER_MODEL (see worker_process), so the two must
        match."""
        return f"OPENARC_WORKER_LOGFILE_{self._model_name.replace('-', '_')}"

    def _worker_log_file(self) -> Optional[str]:
        """The worker's own log file -- the value handed over via
        _worker_logfile_env. A value already set in that env key wins (honoured
        as is, the override); else derive it from OPENARC_LOG_FILE by splicing
        "-worker-<model>" into the base name before ".log" (beside the main log,
        path separators reduced to "-" so the name can't escape its directory).
        None when no main-log location is known."""
        preset = os.environ.get(self._worker_logfile_env(), "").strip()
        if preset:
            return preset
        main_log = os.environ.get("OPENARC_LOG_FILE", "").strip()
        if not main_log:
            return None
        directory, name = os.path.split(main_log)
        base, _ext = os.path.splitext(name)
        return os.path.join(
            directory, f"{base}-worker-{_sanitize_for_filename(self._model_name)}.log"
        )

    def _build_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        # The child must import exactly the code the parent is running,
        # whatever the install mode (editable, wheel, plain cwd): hand over
        # the parent's sys.path and keep the working directory.
        paths = [p for p in sys.path if p]
        cwd = os.getcwd()
        if cwd not in paths:
            paths.insert(0, cwd)
        # Join with the platform's path separator (":" on POSIX, ";" on Windows):
        # reading the Platform keeps this a pure decision that a test can drive for
        # whichever platform, and os.pathsep is the real host's value anyway.
        env["PYTHONPATH"] = self._platform.pathsep.join(paths)
        # The child sizes its stdin reader with this limit BEFORE the LOAD
        # command arrives (the pipe reader is created at startup), so the
        # per-model worker_line_limit travels in the environment, not the
        # protocol.
        env["OPENARC_WORKER_LINE_LIMIT"] = str(self._line_limit)
        # The model name is always passed (the worker's own identity, and we need
        # it for other worker-side things soon too): the worker learns it at
        # startup, before the LOAD message names it, and uses it to read its OWN
        # log-file env var exactly.
        env["OPENARC_WORKER_MODEL"] = self._model_name
        # Also hand the worker its own log file as a per-model env var (a path,
        # value ending in .log); the worker logs all its output there. The
        # supervisor opens no file and forwards nothing; a pre-set value wins.
        worker_log_file = self._worker_log_file()
        if worker_log_file is not None:
            env[self._worker_logfile_env()] = worker_log_file
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
            # Windows only: hide the per-worker console (a visible console would
            # pop a window PER model/worker process), and create no new one.
            # POSIX gets an empty dict -- the child needs no special flags, so the
            # existing Linux/macOS behaviour is byte-for-byte unchanged.
            **platform_support.make_spawn_kwargs(self._platform),
        )
        # Log the started worker's pid from the supervisor so it lands on the
        # server's stdout / openarc.log (the analogue of "Started server process [pid]").
        logger.info(f"[{self._model_name}] inference worker started (pid={self.pid})")
        # Note: the child's stdout/stderr readers keep asyncio's default
        # readline limit, so this side never calls readline() on them --
        # see _read_lines (protocol lines can be very large, see
        # protocol.PROTOCOL_LINE_LIMIT).
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"ovworker-read-{self._model_name}"
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_pump(), name=f"ovworker-stderr-{self._model_name}"
        )

    async def _spawn_and_load(self, timeout: Optional[float]) -> None:
        if self._closed:
            raise proto.RemoteWorkerDeadError("supervisor is closed")
        load_config = self._load_config
        if load_config is None:
            raise proto.RemoteWorkerDeadError("no load config registered")
        self._set_state(self.STATE_STARTING)
        self._fatal_error = None
        self._death_handled = False
        await self._spawn()
        self._start_ping()
        self._load_future = asyncio.get_running_loop().create_future()
        self._set_state(self.STATE_LOADING)
        await self._send(
            proto.encode(
                proto.OP_LOAD, req_id="load", config=load_config.model_dump_json()
            )
        )
        try:
            await asyncio.wait_for(self._load_future, timeout)
        except asyncio.TimeoutError as e:
            raise proto.RemoteWorkerLoadError(
                f"worker did not finish loading within {timeout:.0f}s"
            ) from e
        self._set_state(self.STATE_READY)
        logger.info(f"[{self._model_name}] model loaded (pid={self.pid})")

    # -- sending -----------------------------------------------------------------
    async def _send(self, data: bytes) -> None:
        proc = self._proc
        # A failed write (or an already-exited proc) means the worker died before
        # the reader loop noticed: carry the same will_respawn decision as the
        # other death sites -- defaulting it to False was the one spot that let a
        # still-respawning worker leak a full "respawn budget is exhausted" /
        # "Exception in ASGI application" trace even under no respawncap.
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise proto.RemoteWorkerDeadError(
                "worker process is not running", will_respawn=self._will_respawn()
            )
        try:
            proc.stdin.write(data)
            await proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, ValueError, OSError) as e:
            raise proto.RemoteWorkerDeadError(
                f"cannot write to worker process: {e}", will_respawn=self._will_respawn()
            ) from e

    # -- request API ----------------------------------------------------------------
    async def begin_run(
        self, op: str, gen_config_json: str, request_id: Optional[str] = None
    ) -> Tuple[asyncio.Queue, asyncio.Future]:
        """Send a run request (GENERATE / TRANSCRIBE) and return (item queue,
        result future).

        The queue yields the items the model's run method produced; it always
        ends with the EOF sentinel, after which the result future resolves
        (None) or raises.
        """
        if self._state != self.STATE_READY:
            # Not READY = loading, or respawning / terminated. Mirror the
            # supervisor's own heal-vs-terminal decision: short while restarting,
            # a full trace once it is terminal.
            raise proto.RemoteWorkerDeadError(
                f"inference worker not ready (state={self._state})",
                will_respawn=self._will_respawn(),
            )
        loop = asyncio.get_running_loop()
        req = _PendingRequest(uuid.uuid4().hex)
        req.result = loop.create_future()
        self._active[req.req_id] = req
        await self._send(
            proto.encode(op, req_id=req.req_id, request_id=request_id, gen_config=gen_config_json)
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
                # Graceful terminate: SIGTERM on POSIX, TerminateProcess on
                # Windows (where the graceful and the hard step are the same
                # call -- the ladder below still makes sense as a final resort).
                platform_support.terminate(proc, self._platform)
                try:
                    await asyncio.wait_for(proc.wait(), 5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self._model_name}] worker did not exit after "
                        f"{self._platform.terminate_signal}; forcing kill"
                    )
                    # Hard terminate: SIGKILL on POSIX, TerminateProcess on
                    # Windows (already ported by CPython for both platforms).
                    platform_support.kill(proc, self._platform)
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
    async def _read_lines(self, stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
        """Yield newline-terminated lines of any size up to self._line_limit.

        asyncio's StreamReader.readline() rejects lines longer than the
        reader's limit (64 KiB by default) with ValueError, and the readers
        the subprocess transport creates internally cannot be configured.
        Protocol lines can be far larger (see protocol.PROTOCOL_LINE_LIMIT
        and ModelLoadConfig.worker_line_limit), so read in chunks and split
        on newlines ourselves. read() is the public, version-stable API.
        """
        buffer = b""
        while True:
            chunk = await stream.read(1024 * 1024)
            if not chunk:
                break  # EOF
            buffer += chunk
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                line, buffer = buffer[: newline + 1], buffer[newline + 1:]
                yield line
            if len(buffer) > self._line_limit:
                raise ValueError(
                    f"line from the worker exceeds {self._line_limit} bytes"
                )
        if buffer:
            yield buffer  # final line without a trailing newline

    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            async for line in self._read_lines(proc.stdout):
                try:
                    msg = proto.decode_response(line)
                except proto.ProtocolError as e:
                    logger.warning(f"[{self._model_name}] worker protocol error: {e}")
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                f"[{self._model_name}] worker reader loop crashed; killing the worker"
            )
            # The child may still be alive (e.g. it sent an oversized line);
            # kill it so the proc.wait() below cannot hang.
            self._terminate_process()
        await proc.wait()
        self._close_process_streams()
        self._on_process_exited(proc.returncode)

    async def _stderr_pump(self) -> None:
        # This reads the WORKER child's stderr (the pipe opened at _spawn), never
        # our own: the supervisor/server's own log reaches openarc.log via the
        # separate launch_server dictConfig, independent of this pipe. The worker
        # now puts all of its output in its own file, so this pipe is empty -- we
        # just drain it to EOF (so the worker can't stall on a full pipe, and the
        # unload's pipe wait can finish) and forward nothing.
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            async for line in self._read_lines(proc.stderr):
                pass
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
            req_id = msg.get("req_id")
            if not isinstance(req_id, str):
                return
            req = self._active.get(req_id)
            if req is not None and not (req.result is not None and req.result.done()):
                req.queue.put_nowait(msg.get("item"))
        elif mtype == proto.MSG_DONE:
            self._finish_request(msg.get("req_id"), None)
        elif mtype == proto.MSG_RESULT:
            # Single-result runs (plain-OpenVINO worker protocol): the result
            # payload (e.g. audio samples or a transcription) resolves the
            # request. The GenAI workers never send this.
            self._finish_request(msg.get("req_id"), None, result=msg.get("result"))
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
            message = str(err.get("message") or "worker reported a fatal error")
            self._fatal_error = message
            # The worker is taking itself down on purpose (wedged device): for
            # every in-flight request the worker IS dead, so the failures are
            # RemoteWorkerDeadErrors -- the supervisor owns recovery (respawn
            # within budget, or an unload once it is exhausted), and the
            # registry must not double-act by unloading on the request error.
            exc = proto.RemoteWorkerDeadError(
                message,
                original_type=err.get("type"),
                will_respawn=self._will_respawn(),
            )
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

    def _finish_request(
        self,
        req_id: Optional[str],
        error: Optional[BaseException] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        if req_id is None:
            return
        req = self._active.pop(req_id, None)
        if req is None:
            return
        req.queue.put_nowait(EOF)
        if req.result is not None and not req.result.done():
            if error is None:
                # result is None for streaming runs (DONE); a dict for
                # single-result runs (MSG_RESULT).
                req.result.set_result(result)
            else:
                req.result.set_exception(error)

    def _fail_load(self, exc: BaseException) -> None:
        if self._load_future is not None and not self._load_future.done():
            self._load_future.set_exception(exc)

    def _fail_all_active(self, exc: BaseException) -> None:
        for req_id in list(self._active):
            self._finish_request(req_id, exc)

    # -- death & respawn ------------------------------------------------------------------
    def _will_respawn(self) -> bool:
        """Whether the supervisor will still respawn the worker on this death.

        True while in budget (a positive one not yet spent, or the unlimited
        ``max_respawns <= 0``); otherwise terminal. The single heal-vs-terminal
        decision the rest of the log path keys off, from the supervisor's counters.
        """
        return self._max_respawns <= 0 or self._respawns < self._max_respawns

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
            f"inference worker process exited unexpectedly{detail}{suffix}",
            will_respawn=self._will_respawn(),
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
        # 0 or negative is "no limit": the worker is reloaded every time it dies
        # and is never quarantined (a positive N reintroduces the budget).
        unlimited = self._max_respawns <= 0
        cause = self._fatal_error or "process exited"
        if unlimited or self._respawns < self._max_respawns:
            # Still inside the respawn budget: a HEALING event -- log a single short
            # cause line, no traceback (the worker is merely restarting, in budget).
            self._respawns += 1
            self._set_state(self.STATE_RESTARTING)
            if unlimited:
                logger.warning(
                    f"[{self._model_name}] respawning inference worker "
                    f"(respawn #{self._respawns}; no restart limit); cause: {cause}"
                )
            else:
                logger.warning(
                    f"[{self._model_name}] respawning inference worker "
                    f"({self._respawns}/{self._max_respawns}); cause: {cause}"
                )
            self._respawn_task = asyncio.create_task(
                self._respawn(), name=f"ovworker-respawn-{self._model_name}"
            )
        else:
            # Respawn budget exhausted: the model is now TERMINAL -- a full
            # traceback is allowed (emitted by the next request that sees the dead worker).
            self._set_state(self.STATE_DEAD)
            logger.error(
                f"[{self._model_name}] respawn budget exhausted "
                f"({self._respawns}/{self._max_respawns}); worker is dead; "
                f"cause: {cause}"
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
            if self._max_respawns <= 0:
                # No restart limit: a failed reload is never given up on -- keep
                # trying to bring the worker back (backed off) until it comes up
                # or the model is explicitly unloaded.
                logger.error(
                    f"[{self._model_name}] worker respawn failed: {e}; "
                    f"will keep trying to reload it (no restart limit)"
                )
                self._set_state(self.STATE_RESTARTING)
                self._respawn_task = asyncio.create_task(
                    self._respawn(), name=f"ovworker-respawn-{self._model_name}"
                )
                return
            logger.error(f"[{self._model_name}] worker respawn failed: {e}")
            self._set_state(self.STATE_DEAD)
            if self._on_dead is not None:
                await self._fire_on_dead()

    async def _fire_on_dead(self) -> None:
        on_dead = self._on_dead
        if on_dead is None:
            return
        try:
            await on_dead()
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
        # Route the emergency terminate through the platform seam: SIGKILL on
        # POSIX, TerminateProcess on Windows; the helper swallows the "already
        # gone" case (a child that exited on its own before we got here).
        platform_support.kill(proc, self._platform)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            logger.debug(f"[{self._model_name}] worker state {self._state} -> {state}")
            self._state = state
