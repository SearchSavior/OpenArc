"""Platform-specific process handling for the inference worker machinery.

The supervisor/worker design is *otherwise* identical on POSIX (Linux/macOS) and
Windows: the worker is one `sys.executable -c ...` child per model, it speaks the
same one-JSON-line-over-stdin/out protocol, it redirects its own stderr into its
own log file (``os.open``/``os.dup2`` are cross-platform), and it exits with the
same exit codes (``os._exit``/``sys.exit``). No fork, no process groups, no
POSIX signals are relied on -- confirmed by grepping the codebase.

Only a few OS primitives differ, and that is all this module owns:

  * **Console / stdin.** This module adds *no* spawn control on either host
    (``hide_console=False``; ``make_spawn_kwargs`` returns ``{}``). The real
    breakage was in the worker, not the spawn: on Windows the worker reads commands
    by ``connect_read_pipe``-ing its inherited ``sys.stdin``, which the proactor
    then registers with its IOCP via ``CreateIoCompletionPort`` -- and that fails
    ``[WinError 6] The handle is invalid`` on the inherited (non-overlapped) stdin
    pipe, so the worker reads nothing and the 30s load budget expires (a plain
    ``os.read(0)`` on the same handle works). Fixed in ``worker_process``: on
    Windows the worker reads ``stdin`` with a blocking ``os.read`` on a daemon
    thread feeding the StreamReader (POSIX keeps ``connect_read_pipe``).
  * **Termination.** On POSIX the supervisor escalates ``SIGTERM`` (terminate)
    then ``SIGKILL`` (kill); on Windows asyncio routes BOTH ``Process.terminate()``
    and ``Process.kill()`` to ``_winapi.TerminateProcess`` -- so the whole ladder
    collapses toward one hard terminate. The *graceful* path is the protocol's
    ``UNLOAD`` message (the child exits 0 itself); terminate/kill are only the
    fallback after a timeout, and ``Process.terminate()``/``Process.kill()`` are
    already ported for both platforms by CPython. This module just centralises the
    call (and the Platform key on which a single test suite can assert both hosts).
  * **PYTHONPATH joining.** ``os.pathsep`` is ``":"`` on POSIX and ``";"`` on
    Windows; the supervisor already joins with ``os.pathsep`` (correct on any
    host) -- we route it through ``Platform.pathsep`` so the *decision* is pure
    and testable regardless of which host the test happens to run on.

Why split into DECISION vs EXECUTION
------------------------------------
The whole point of the split is to make the Windows path *testable on a POSIX
box*, since the project has no Windows environment:

  * ``describe_spawn_kwargs`` / ``Platform.pathsep`` / ``build_*`` are *pure*
    functions of a host-independent ``Platform`` descriptor. They can be called
    with BOTH ``POSIX`` and ``WINDOWS`` on any host and asserted -- this is what
    the unit tests do, so the Windows *decision* (hide the console, ``";"`` path
    separator, which terminate strategy) is checked without ever touching a real
    Windows primitive.
  * ``make_spawn_kwargs`` is the one step that translates the pure description
    into real handles/flags. The Windows-only symbols (``subprocess.STARTUPINFO``,
    ``STARTF_USESHOWWINDOW``, ``SW_HIDE``, ``CREATE_NO_WINDOW``) exist *only* on a
    Windows host, so that import is guarded by ``os.name == "nt"``; if a caller
    asks for the Windows spawn from a POSIX host it gets a clear ``RuntimeError``
    instead of a cryptic ``AttributeError``. The *actual* Windows spawn therefore
    only ever runs on a Windows host -- which the ``windows-latest`` CI job
    provides (the same stub lifecycle unit tests then auto-exercise the real
    Windows spawn/terminate, because the supervisor uses ``current_platform()``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

# The host we actually run on (``os.name`` is ``"nt"`` on Windows, ``"posix"``
# everywhere this project supports).
IS_WINDOWS = os.name == "nt"
IS_POSIX = os.name == "posix"
OS_NAME = os.name


@dataclass(frozen=True)
class Platform:
    """A host-independent description of a platform's process behaviour.

    This is deliberately free of any OS-specific handles/symbols: it only records
    *what* the platform wants to do, so the pure decision functions (and the tests
    over them) can run on any host. ``make_spawn_kwargs`` performs the one
    host-specific translation from this description into real kwargs.

    Attributes:
        name: ``"posix"`` or ``"windows"``.
        pathsep: separator to join a Python path for the child's ``PYTHONPATH``
            (``":"`` on POSIX, ``";"`` on Windows) -- read from ``os.pathsep`` for
            the real host so nothing doubles up with the OS.
        hide_console: whether to spawn the child with a hidden console (Windows
            only; a visible one would pop a window per model/worker process).
        uses_job_object: reserved -- kept as a flag for a future where a worker
            spawns further children and we'd want to reap the whole tree. OpenVINO
            inference runs in-process (threads), so a worker never branches into a
            child tree, and no job object is needed today.
        terminate_signal: how the hard-terminate fallback maps (see module
            docstring); purely informational for log messages / the tests.
    """

    name: str
    pathsep: str
    hide_console: bool
    uses_job_object: bool = False
    terminate_signal: str = "sigterm_then_sigkill"


# Host-independent descriptors for the two platforms. ``pathsep`` is HARDCODED
# (":" on POSIX, ";" on Windows) -- NOT read from os.pathsep -- so a test running
# on either host can drive the *other* platform's decisions (e.g. assert the
# windowed spawn / ";"-joined PYTHONPATH) without a real Windows box. The real
# ``os.pathsep`` is used only by ``current_platform()`` below, which the
# supervisor actually runs on (so production on any host is always correct).
# ``describe_spawn_kwargs``/``make_spawn_kwargs`` key off ``hide_console`` only,
# so these fixed separators never affect the spawn resolution.
POSIX = Platform(
    name="posix",
    pathsep=":",
    hide_console=False,
    terminate_signal="sigterm_then_sigkill",
)

WINDOWS = Platform(
    name="windows",
    pathsep=";",
    # See the module docstring: we no longer hide the per-worker console. Forcing a
    # hidden console clobbered the proactor's STARTF_USESTDHANDLES wiring and left
    # the child's stdin an invalid handle (WinError 6) -- an invalid, silent child.
    hide_console=False,
    uses_job_object=False,
    terminate_signal="terminate_process",
)


def current_platform() -> "Platform":
    """The ``Platform`` for the host this process is actually running on.

    Uses the real ``os.pathsep``/``os.name`` so the production path on any host is
    correct; the fixed module-level ``POSIX``/``WINDOWS`` are exposed separately so
    a test can drive the *other* platform's decisions from whatever host it runs on.
    """
    if os.name == "nt":
        return Platform(
            name="windows",
            pathsep=os.pathsep,
            hide_console=False,
            uses_job_object=False,
            terminate_signal="terminate_process",
        )
    return Platform(
        name="posix",
        pathsep=os.pathsep,
        hide_console=False,
        terminate_signal="sigterm_then_sigkill",
    )


# -- pure spawn *description* (host-independent: asserted by the unit tests) --------


def describe_spawn_kwargs(platform: "Platform") -> Dict[str, Any]:
    """Host-independent description of the extra kwargs needed to spawn a worker
    child, expressed as plain data (names/booleans, NOT live handles) so it can be
    asserted on any host.

    With the current decision (hidden console *disabled* -- see the module
    docstring) BOTH platforms need nothing extra, so this returns ``{}`` and the
    worker's stdin/stdout/stderr are wired straight by asyncio's proactor. The dict
    shape below is retained only as the documented form a caller would receive if a
    *proactor-safe* console control were ever re-introduced; today no ``Platform``
    sets ``hide_console`` True, so that branch is unreachable.
    """
    if not platform.hide_console:
        return {}
    return {
        "creationflags": "CREATE_NO_WINDOW",
        "startupinfo_dwFlags": "STARTF_USESHOWWINDOW",
        "startupinfo_wShowWindow": "SW_HIDE",
    }


def make_spawn_kwargs(platform: "Platform") -> Dict[str, Any]:
    """Translate a ``Platform``'s spawn description into the real kwargs for
    ``asyncio.create_subprocess_exec``.

    Both platforms currently return ``{}``: with the hidden console disabled (see
    the module docstring) the POSIX AND the Windows child need no special spawn
    flags at all, so asyncio's proactor wires the child's stdin/stdout/stderr
    itself and the child's stdin ends up a valid handle (the thing a forced hidden
    console broke). The Windows-only symbol-build and the actionable off-host error
    below are retained as the documented shape for if/when a *proactor-safe*
    console control is re-introduced.

    Guarding the Windows-only symbols: ``subprocess.STARTUPINFO`` /
    ``STARTF_USESHOWWINDOW`` / ``SW_HIDE`` / ``CREATE_NO_WINDOW`` are defined
    *only* on a Windows interpreter, so the corresponding import/lookup happens
    strictly after an ``os.name == "nt"`` check. If a caller asks for the Windows
    spawn from a POSIX host (e.g. a test that forgot to gate on the host), this
    raises a clear, actionable ``RuntimeError`` rather than a cryptic
    ``AttributeError`` deep inside ``create_subprocess_exec``.
    """
    desc = describe_spawn_kwargs(platform)
    if not desc:
        # POSIX (or any platform that needs no extra flags): nothing to add.
        return {}
    # A console-hiding spawn (Windows). The Windows-only subprocess symbols are
    # only looked up on a Windows host; any other host gets an actionable error
    # instead of a bare AttributeError from an incomplete platform module.
    if os.name != "nt":
        raise RuntimeError(
            f"cannot construct {platform.name!r} console-hiding spawn kwargs on host "
            f"os.name={os.name!r}: the Windows STARTUPINFO/CREATE_NO_WINDOW symbols "
            f"only exist on a Windows interpreter. The {platform.name!r} spawn runs "
            f"there (see the windows-latest CI); on a POSIX box assert "
            f"describe_spawn_kwargs({platform.name!r}) instead of calling this."
        )
    import subprocess as _subprocess

    startupinfo = _subprocess.STARTUPINFO()
    # The child inherits our handles via asyncio's STARTF_USESTDHANDLES (added by
    # Popen for the stdin/stdout/stderr pipes); we OR in USESHOWWINDOW and hide,
    # and request that no NEW console be created, so a worker never pops a window.
    startupinfo.dwFlags |= _subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = _subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": _subprocess.CREATE_NO_WINDOW,
    }


# -- termination (asyncio Process; the graceful UNLOAD is done in the supervisor) --

# A duck-typed stand-in for asyncio.subprocess.Process: only the two methods we call.
Terminable = Any


def terminate(proc: "Terminable", platform: "Platform") -> None:
    """Soft-terminate the worker child (best effort; the supervisor calls this in
    the unload ladder after the graceful ``UNLOAD`` message times out, and the
    result is re-checked by awaiting ``proc.wait()``).

    On POSIX this is ``SIGTERM``; on Windows asyncio routes ``Process.terminate()``
    to ``TerminateProcess``, so for Windows the graceful-then-hard ladder the
    supervisor runs is effectively a single hard terminate. Swallows the "already
    gone" cases (a child that exited on its own before we got here).
    """
    try:
        proc.terminate()
    except (ProcessLookupError, OSError):
        pass


def kill(proc: "Terminable", platform: "Platform") -> None:
    """Hard-terminate the worker child (the last step of the unload ladder).

    ``Process.kill()`` is already ported by CPython for both platforms (``SIGKILL``
    on POSIX, ``TerminateProcess`` on Windows), so the call is identical -- this
    wrapper just keeps the platform key in one place and swallows the "already
    gone" cases so the supervisor's ``await proc.wait()`` does the real ordering.
    """
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
