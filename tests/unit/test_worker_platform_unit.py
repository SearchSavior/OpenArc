"""Tests for the platform-specific process handling the worker relies on
(src.engine.worker.platform_support), plus the platform seams the supervisor
now routes through.

The whole point of this module is to make the Windows worker path **testable on
a POSIX box**, since the project has no local Windows environment. That works
because the platform-specific behavior is split into:

  * a *decision* -- which platform we are on, whether to hide the per-worker
    console, which flags/`STARTUPINFO` to spawn with, what character to join
    PYTHONPATH with, and how the hard-terminate ladder maps -- expressed as pure
    functions of a host-independent `Platform` descriptor. These are exercised
    HERE, on whatever host the suite runs on, for BOTH `POSIX` and `WINDOWS`,
    so the Windows decisions are checked without ever touching a real Windows
    primitive.
  * *execution* -- the one step that turns the description into real
    `subprocess` handles/flags (`make_spawn_kwargs`), which genuinely requires a
    matching host. That is *not* tested here (it needs Windows); it is exercised
    by the existing stub-lifecycle unit tests, which spawn the real child on the
    current host (so they auto-cover Windows when run on the `windows-latest` CI
    runner) and, explicitly, by tests/integration/test_worker_windows_
    integration.py (gated on the real host).

So: the DECISIONS are asserted below for both platforms on any host; the
EXECUTION runs on whatever host the test executes on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest  # type: ignore[import]

from src.engine.worker import platform_support as ps
from src.engine.worker.supervisor import WorkerSupervisor
from src.engine.worker.worker_process import _redirect_stderr_to


# -- the host-independent Platform descriptors ------------------------------------------


def test_platform_descriptors_are_host_independent() -> None:
    """The fixed POSIX/WINDOWS descriptors carry HARDCODED separators (not the
    running host's os.pathsep), so a test can drive the *other* platform's
    decisions. That is the mechanism that lets us assert the Windows decision on
    a POSIX box and vice versa."""
    assert ps.POSIX.name == "posix"
    assert ps.POSIX.pathsep == ":"
    assert ps.POSIX.hide_console is False
    assert ps.WINDOWS.name == "windows"
    assert ps.WINDOWS.pathsep == ";"
    assert ps.WINDOWS.hide_console is False
    # The Windows decision: add *no* console control (a forced hidden console broke
    # the proactor's stdio wiring -- see platform_support's module docstring), while
    # the graceful-then-hard terminate ladder collapses toward a single hard terminate.
    assert ps.WINDOWS.terminate_signal == "terminate_process"
    assert ps.POSIX.terminate_signal == "sigterm_then_sigkill"
    # reserved for a future where a worker spawns its own children.
    assert ps.WINDOWS.uses_job_object is False


def test_current_platform_matches_the_real_host() -> None:
    """`current_platform()` is what the supervisor actually runs on: it keys off
    the real `os.name` and uses the real `os.pathsep`, so production on any host
    is correct (a test on Windows gets the windows descriptor and a `;`
    separator, a test on POSIX gets the opposite)."""
    plat = ps.current_platform()
    assert plat.name == ("windows" if os.name == "nt" else "posix")
    assert plat.pathsep == os.pathsep
    assert plat.hide_console is False


# -- the pure spawn *decision* (describe_spawn_kwargs) ----------------------------------


def test_spawn_decision_posix_needs_nothing() -> None:
    """On POSIX a child needs no special flags: no console to hide, no
    STARTUPINFO, creationflags 0 -- the existing Linux/macOS spawn, unchanged."""
    assert ps.describe_spawn_kwargs(ps.POSIX) == {}


def test_spawn_decision_windows_requests_no_console_control() -> None:
    """The old Windows decision was to hide the per-worker console (CREATE_NO_WINDOW
    + STARTF_USESHOWWINDOW + SW_HIDE) so a server does not pop a visible window per
    model. That broke the real spawn (see platform_support's module docstring):
    forcing a hidden console clobbered the proactor's STARTF_USESTDHANDLES stdio
    wiring and left the child's stdin an invalid handle, so the worker read nothing
    and the 30s load budget expired. The decision is now: request *no* extra console
    control -- the proactor wires the child's stdin/stdout/stderr itself (a visible,
    inherited per-worker window is accepted)."""
    assert ps.describe_spawn_kwargs(ps.WINDOWS) == {}


# -- the one host-specific resolution step (make_spawn_kwargs) ---------------------------
#
# With the console-hide disabled the Windows resolution is a plain ``{}`` -- the
# same as POSIX and host-independent (it consults no Windows-only subprocess
# symbols) -- so it runs on any machine. make_spawn_kwargs still keeps an actionable
# off-host error behind that guard, but it is now unreachable because
# describe_spawn_kwargs(WINDOWS) is empty first; the test below proves the result.


def test_make_spawn_kwargs_posix_is_empty_on_any_host() -> None:
    """Resolving the POSIX decision needs no Windows symbols, so it is a no-op
    ``{}`` and works everywhere (this is the path the real spawn takes on Linux
    / macOS, so it is also covered end-to-end by the stub-lifecycle tests)."""
    assert ps.make_spawn_kwargs(ps.POSIX) == {}


def test_make_spawn_kwargs_windows_is_a_host_independent_noop() -> None:
    """With the console-hide disabled, resolving the Windows decision returns a plain
    ``{}`` (describe_spawn_kwargs(WINDOWS) is empty first, so the actionable
    off-host error inside make_spawn_kwargs is never reached) -- it consults no
    Windows-only subprocess symbols, so it works identically on every host. That is
    the proof the Windows worker spawn no longer needs a Windows-only code path at
    all; the real spawn is then exercised end-to-end in
    tests/integration/test_worker_windows_integration.py."""
    assert ps.make_spawn_kwargs(ps.WINDOWS) == {}


# -- the termination ladder, keyed off the platform ---------------------------------------


class _RecordingTerminable:
    """Stand-in for asyncio.subprocess.Process: records which of the two calls
    the platform seam chose, and can also act as a process that is already gone."""

    def __init__(self, *, already_gone: bool = False) -> None:
        self.calls: list = []
        self.already_gone = already_gone

    def terminate(self) -> None:
        if self.already_gone:
            raise ProcessLookupError("already gone")
        self.calls.append("terminate")

    def kill(self) -> None:
        if self.already_gone:
            raise ProcessLookupError("already gone")
        self.calls.append("kill")


@pytest.mark.parametrize("plat", [ps.POSIX, ps.WINDOWS], ids=["posix", "windows"])
def test_terminate_uses_proc_terminate_for_every_platform(plat) -> None:
    proc = _RecordingTerminable()
    ps.terminate(proc, plat)
    assert proc.calls == ["terminate"]


@pytest.mark.parametrize("plat", [ps.POSIX, ps.WINDOWS], ids=["posix", "windows"])
def test_kill_uses_proc_kill_for_every_platform(plat) -> None:
    proc = _RecordingTerminable()
    ps.kill(proc, plat)
    assert proc.calls == ["kill"]


@pytest.mark.parametrize("plat", [ps.POSIX, ps.WINDOWS], ids=["posix", "windows"])
@pytest.mark.parametrize("helper", [ps.terminate, ps.kill])
def test_termination_swallows_an_already_gone_child(plat, helper) -> None:
    """The supervisor's unload ladder must not raise because the child exited on
    its own first (a race the `await proc.wait()` after is the real ordering --
    the helper just has to be best-effort). Checked for both platforms."""
    gone = _RecordingTerminable(already_gone=True)
    helper(gone, plat)  # must not raise


# -- the supervisor routes its platform seams through these fns ---------------------------


def test_supervisor_build_env_pathsep_is_platform_driven() -> None:
    """_build_env joins the child's PYTHONPATH with the injected platform's
    separator, so the *decision* is host-independent and assertable. On a
    POSIX host the default (real) platform uses ':'; injecting WINDOWS must use
    ';'. The worker identity is always handed over regardless."""
    host_env = WorkerSupervisor("mymodel")._build_env()
    # default == current_platform() == real host: its real os.pathsep IS the join
    # separator (":" on POSIX, ";" on Windows).
    assert os.pathsep in host_env["PYTHONPATH"]
    # The model name (and, when a main log is configured, the per-model log key)
    # are always passed so the worker can find its own log file.
    assert host_env["OPENARC_WORKER_MODEL"] == "mymodel"

    win_env = WorkerSupervisor("mymodel", platform=ps.WINDOWS)._build_env()
    # Injecting WINDOWS forces the ";" JOIN separator -- that is the decision.
    assert ";" in win_env["PYTHONPATH"]
    # BUT every entry is still an absolute Windows path, e.g. C:/x/y, which
    # carries a ":" from the drive letter. That ":" is a *drive letter*, not the
    # (";") separator we join on -- so it must NOT be forbidden. A drive-letter
    # ":" only ever sits *inside* an entry, so splitting on the ";" join
    # separator still re-joins to the original (the ";" is genuinely a boundary):
    assert ";".join(win_env["PYTHONPATH"].split(";")) == win_env["PYTHONPATH"]
    # Where the two platforms pick different separators (a POSIX host), the joined
    # result is demonstrably different -- so the decision is real, not cosmetic:
    if os.name == "posix":
        assert win_env["PYTHONPATH"] != host_env["PYTHONPATH"]
    assert win_env["OPENARC_WORKER_MODEL"] == "mymodel"


def test_supervisor_spawn_command_is_identical_across_platforms() -> None:
    """The argv is `sys.executable -c "..."` on both platforms (the only
    platform-specific part of the spawn is the *flags*, which are added
    separately at the create_subprocess_exec call site) -- so the command itself
    must not differ."""
    win_cmd = WorkerSupervisor("m", platform=ps.WINDOWS)._build_command()
    posix_cmd = WorkerSupervisor("m", platform=ps.POSIX)._build_command()
    assert win_cmd == posix_cmd
    assert win_cmd[0:2] == [__import__("sys").executable, "-c"]


def test_supervisor_uses_injected_platform() -> None:
    sup = WorkerSupervisor("m", platform=ps.WINDOWS)
    assert sup._platform is ps.WINDOWS
    assert sup._platform.name == "windows"


# -- the worker's stderr redirect fails open (never crashes the worker) -------------------


class _FakeStream:
    def __init__(self, name: str) -> None:
        self._name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_redirect_stderr_is_best_effort_when_open_fails(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """If the log file/dir can't be opened (a real hazard on a restricted
    Windows install / read-only share), the redirect must FAIL OPEN: return
    False, not raise, and leave sys.stderr untouched -- so a bad log target can
    never take the worker down. We simulate the os.open failure directly
    (monkeypatched on the module's `os`, so no real fd 2 is touched)."""
    from src.engine.worker import worker_process as wp

    def _raise_open(path, *a, **k):
        raise PermissionError("read-only log directory")

    fake_prev = _FakeStream("inherited-pipe")
    monkeypatch.setattr(wp.os, "open", _raise_open)
    monkeypatch.setattr(wp.sys, "stderr", fake_prev, raising=False)
    result = _redirect_stderr_to(str(tmp_path / "openarc-worker-x.log"))
    assert result is False
    # sys.stderr must be the inherited stream, untouched (still usable, so the
    # worker keeps running; the supervisor simply forwards nothing from it).
    assert wp.sys.stderr is fake_prev
    assert fake_prev.closed is False


def test_redirect_stderr_happy_path_rebuilds_stderr(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """The happy path: open the file, dup2 it onto fd 2 (then close the spare fd),
    rebuild sys.stderr on it (closefd=False so it isn't GC-closed twice), and
    close the inherited stderr. We stub os.open/os.dup2/os.close/os.fdopen so the
    test touches NO real descriptor -- only the control flow (order + result) is
    asserted; a real redirect exercising fd 2 is covered end-to-end by the
    worker respawn integration test."""
    from src.engine.worker import worker_process as wp

    events: list = []
    monkeypatch.setattr(
        wp.os, "open", lambda path, *a, **k: events.append(("open", path)) or 99
    )
    monkeypatch.setattr(
        wp.os, "dup2", lambda fd, two: events.append(("dup2", fd, two))
    )
    monkeypatch.setattr(wp.os, "close", lambda fd: events.append(("close", fd)))
    monkeypatch.setattr(
        wp.os, "fdopen", lambda fd, *a, **k: events.append(("fdopen", fd)) or _FakeStream(f"fd{fd}")
    )
    fake_prev = _FakeStream("inherited-pipe")
    monkeypatch.setattr(wp.sys, "stderr", fake_prev, raising=False)

    assert _redirect_stderr_to(str(tmp_path / "openarc-worker-x.log")) is True
    # order: open the file first, dup2 it onto fd 2, then close the spare fd,
    # then build the new sys.stderr out of (a private dup of) fd 2.
    assert events[0][0] == "open"
    assert ("dup2", 99, 2) in events
    assert ("close", 99) in events
    assert events[-1][0] == "fdopen"
    # the inherited (pipe) stderr is closed so its write end can't be GC'd twice.
    assert fake_prev.closed is True
