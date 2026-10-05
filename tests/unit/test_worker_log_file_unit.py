"""The worker's OWN log file.

Each worker's full logging (and any native C-level stderr) is redirected, at
startup in the worker, into a file beside the main log -- named after
`<base>.log` with `-worker-<model>` spliced in before the `.log`, so
`openarc.log` becomes `openarc-worker-<model>.log` in the same directory. The
supervisor hands that path to each worker in a per-model env var --
OPENARC_WORKER_LOGFILE_<model>, the dash in the model name written as an
underscore, one key per worker; a value already set there pins the file instead.
The worker also gets its model name, always, in OPENARC_WORKER_MODEL, so at
startup it looks up its OWN key exactly (not by a prefix scan, which a sibling
model's inherited key could fool); `_configure_logging` then redirects into it.
Only the worker ever writes to the file: the supervisor opens no file and merely
drains the pipe (forwarding nothing), so `openarc.log` keeps only the supervisor's
view of the worker.

These are the deterministic, subprocess-free halves: the filename derivation,
the env plumbing (the supervisor's _build_env passes the model name and the
per-model file key; the worker's _worker_logfile_from_env reads the latter by
exact name), and the pump now simply draining the quiet pipe. The real redirect +
a real worker writing to the file (and openarc.log staying clean of it) is
exercised end to end by tests/integration/test_worker_respawn_integration.py.

Run with:  pytest -W ignore tests/unit/test_worker_log_file_unit.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import pytest  # type: ignore[import]

from src.engine.worker.supervisor import WorkerSupervisor, _sanitize_for_filename
from src.engine.worker.worker_process import _worker_logfile_from_env


# -- _sanitize_for_filename: a model name is used verbatim, but must never escape
#    its directory (a path separator would create a subdirectory of the log dir)

def test_sanitize_keeps_dots_and_dashes() -> None:
    # Dots and dashes in a model name (e.g. "Qwen3.8.5", "Qwen3.8-1.8b") survive
    # unchanged -- the name is spliced in verbatim, per the naming rule.
    assert _sanitize_for_filename("Qwen3.8.5") == "Qwen3.8.5"
    assert _sanitize_for_filename("Qwen3.8-1.8b") == "Qwen3.8-1.8b"


def test_sanitize_neutralises_path_separators() -> None:
    # But "/" and "\" would let the file leave the log directory, so reduce them
    # to "-"; NUL is neutralised too.
    assert _sanitize_for_filename("a/b") == "a-b"
    assert _sanitize_for_filename("a\\b") == "a-b"
    assert _sanitize_for_filename("a\x00b") == "a-b"


# -- _worker_log_file: the value handed over in OPENARC_WORKER_LOGFILE_<model> --
#    a value already set there wins (the override); else derive
#    <base>-worker-<model>.log beside OPENARC_LOG_FILE. The env KEY still keeps
#    the model name with "-" -> "_", but the file name keeps it verbatim.

def test_worker_log_file_beside_main_log(monkeypatch: "pytest.MonkeyPatch") -> None:
    # No directory component: the file is "<base>-worker-<model>.log" next to the
    # main "<base>.log" (in the same -- here, the current -- directory).
    monkeypatch.setenv("OPENARC_LOG_FILE", "openarc.log")
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_Qwen3.8.5", raising=False)
    sup = WorkerSupervisor("Qwen3.8.5")
    assert sup._worker_log_file() == "openarc-worker-Qwen3.8.5.log"


def test_worker_log_file_keeps_the_main_log_directory(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # A directory component is (re)attached around the model-part splice, so the
    # worker file lands in the SAME directory as the main log. The EXPECTED
    # value is built with os.path.join -- i.e. the host's separator ("\" on a
    # Windows box) -- exactly the way the code builds it, so this holds on either
    # host. The former hard-coded "/var/log/..." read back on Windows as
    # "/var/log\\openarc-worker-mymodel.log"; a drive letter (e.g. C:/x/y) is a
    # perfectly valid Windows path, so the point is the *shape follows
    # os.path.join*, not that a path can never contain a colon.
    main_log = "/var/log/openarc.log"
    monkeypatch.setenv("OPENARC_LOG_FILE", main_log)
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_mymodel", raising=False)
    sup = WorkerSupervisor("mymodel")
    result = sup._worker_log_file()
    # The named intent: same directory as the main log, the model part spliced in.
    # Both derived on the host, so the separator matches whatever os.path produces.
    directory, _name = os.path.split(main_log)
    assert result == os.path.join(directory, "openarc-worker-mymodel.log")
    assert os.path.dirname(result) == os.path.dirname(main_log)
    assert os.path.basename(result) == "openarc-worker-mymodel.log"


def test_worker_log_file_is_per_model(monkeypatch: "pytest.MonkeyPatch") -> None:
    # Distinct models get distinct files: each model owns a process and a log.
    # The expected paths are built with os.path.join (the host separator -- "\"
    # on Windows) the same way the code builds them, so this holds on either host;
    # the former hard-coded "/var/log/..." read back on Windows as
    # "/var/log\\openarc-worker-alpha.log" (a drive letter, e.g. C:/x/y, is a
    # valid Windows path, so the separator must simply follow os.path.join).
    main_log = "/var/log/openarc.log"
    monkeypatch.setenv("OPENARC_LOG_FILE", main_log)
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_alpha", raising=False)
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_beta", raising=False)
    a = WorkerSupervisor("alpha")._worker_log_file()
    b = WorkerSupervisor("beta")._worker_log_file()
    directory, _name = os.path.split(main_log)
    assert a == os.path.join(directory, "openarc-worker-alpha.log")
    assert b == os.path.join(directory, "openarc-worker-beta.log")
    assert a != b
    # Same directory, only the model part of the name differs -- so neither can be
    # mistaken for the other's file. (Basename is separator-free, holds on any host.)
    assert os.path.dirname(a) == os.path.dirname(b)
    assert os.path.basename(a) == "openarc-worker-alpha.log"
    assert os.path.basename(b) == "openarc-worker-beta.log"


def test_worker_log_file_honours_pre_set_override(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # A value already set in the per-model key wins over the derived path. The
    # model name has a dash, so the KEY (and thus the pre-set var) uses the dash
    # swapped for an underscore -- the mapping the supervisor keys on.
    monkeypatch.setenv("OPENARC_LOG_FILE", "/var/log/openarc.log")
    monkeypatch.setenv(
        "OPENARC_WORKER_LOGFILE_my_model", "/tmp/anything/openarc-worker-x.log"
    )
    sup = WorkerSupervisor("my-model")
    assert sup._worker_log_file() == "/tmp/anything/openarc-worker-x.log"


def test_worker_log_file_none_without_main_log(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # No main-log location and no pre-set value: return None, and the spawn env
    # carries no worker-log key (there is now nothing to hand over).
    monkeypatch.delenv("OPENARC_LOG_FILE", raising=False)
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_mymodel", raising=False)
    sup = WorkerSupervisor("mymodel")
    assert sup._worker_log_file() is None


# -- _build_env: passes the model name, and the per-model log-file key ---------

def test_build_env_passes_worker_log_file(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENARC_LOG_FILE", str(tmp_path / "openarc.log"))
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_mymodel", raising=False)
    sup = WorkerSupervisor("mymodel")
    env = sup._build_env()
    # The model name is always passed (the worker's own identity) ...
    assert env["OPENARC_WORKER_MODEL"] == "mymodel"
    # ... and the derived file is handed to the child under the per-model key.
    assert env["OPENARC_WORKER_LOGFILE_mymodel"] == str(
        tmp_path / "openarc-worker-mymodel.log"
    )


def test_build_env_passes_pre_set_override(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    # A pre-set value -- in the per-model key (dash -> underscore) -- is copied
    # through to the child, not recomputed from OPENARC_LOG_FILE.
    monkeypatch.setenv("OPENARC_LOG_FILE", str(tmp_path / "openarc.log"))
    monkeypatch.setenv(
        "OPENARC_WORKER_LOGFILE_my_model", "/tmp/anything/openarc-worker-x.log"
    )
    sup = WorkerSupervisor("my-model")
    env = sup._build_env()
    assert env["OPENARC_WORKER_MODEL"] == "my-model"
    assert env["OPENARC_WORKER_LOGFILE_my_model"] == (
        "/tmp/anything/openarc-worker-x.log"
    )


def test_build_env_omits_worker_log_file_without_main_log(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # No main log and no pre-set value: the env carries no worker-log key ...
    monkeypatch.delenv("OPENARC_LOG_FILE", raising=False)
    monkeypatch.delenv("OPENARC_WORKER_LOGFILE_mymodel", raising=False)
    sup = WorkerSupervisor("mymodel")
    env = sup._build_env()
    assert "OPENARC_WORKER_LOGFILE_mymodel" not in env
    # ... but the model name is still passed (the worker always knows who it is).
    assert env["OPENARC_WORKER_MODEL"] == "mymodel"


# -- worker side: the worker reads its OWN log-file key by exact name ----------
#    (it knows its model name from OPENARC_WORKER_MODEL at startup, so a sibling
#    model's key inherited from the server's environment can't be mistaken for it)

def test_worker_reads_exact_key_for_its_model(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    # Two models, two keys (as a multi-model server would pin): the worker, told
    # its name, reads ONLY its own key.
    monkeypatch.setenv("OPENARC_WORKER_LOGFILE_mymodel", str(tmp_path / "mine.log"))
    monkeypatch.setenv(
        "OPENARC_WORKER_LOGFILE_othermodel", str(tmp_path / "other.log")
    )
    assert _worker_logfile_from_env("mymodel") == str(tmp_path / "mine.log")


def test_worker_logfile_key_rewrites_dash(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    # The dash becomes an underscore in the key (matching the supervisor), so a
    # hyphenated model name still resolves to its own key.
    monkeypatch.setenv("OPENARC_WORKER_LOGFILE_my_model", str(tmp_path / "x.log"))
    assert _worker_logfile_from_env("my-model") == str(tmp_path / "x.log")


def test_worker_no_file_when_none_handed(monkeypatch: "pytest.MonkeyPatch") -> None:
    # We know our name, but the supervisor handed it no file -> None (so we log
    # to our own stderr and the supervisor forwards nothing).
    for k in list(os.environ):
        if k.startswith("OPENARC_WORKER_LOGFILE_"):
            monkeypatch.delenv(k, raising=False)
    assert _worker_logfile_from_env("mymodel") is None
    # ... and with no name at all there is nothing to key on, so the helper stays
    # safe without one (the supervisor supplies a name, but this must not raise).
    assert _worker_logfile_from_env("") is None


# -- _stderr_pump: it only drains the pipe; it never forwards into the main log --

class _FakeStream:
    """A stand-in for asyncio's StreamReader: yield the queued lines, then EOF."""

    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeProc:
    """A stand-in for the worker subprocess: `stderr` is the (quiet) pipe the
    pump drains, `pid`/`returncode` are what the pump references."""

    returncode: Optional[int] = None

    def __init__(self, stderr_chunks: List[bytes]) -> None:
        self.pid = 777
        self.stderr = _FakeStream(stderr_chunks)


def test_stderr_pump_drains_without_forwarding(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: Path,
    caplog: "pytest.LogCaptureFixture",
) -> None:
    # Whatever (accidentally) reaches the pipe is drained -- never echoed into the
    # main log, since the worker already wrote it to its own file.
    monkeypatch.setenv("OPENARC_LOG_FILE", str(tmp_path / "openarc.log"))
    sup = WorkerSupervisor("mymodel")
    sup._build_env()  # a real file is handed over, as in a real run
    sup._proc = _FakeProc([b"worker stderr line\n"])
    caplog.set_level(logging.INFO)
    asyncio.run(sup._stderr_pump())
    lines = [
        r.getMessage() for r in caplog.records if r.name == "src.engine.worker.supervisor"
    ]
    assert "worker stderr line" not in " ".join(lines), lines
