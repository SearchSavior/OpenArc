"""End-to-end test of the *Windows-specific* spawn of an inference worker, run
only on a Windows host.

The *decision* that the Windows path exists (add *no* console control, join
PYTHONPATH with ';', terminate via TerminateProcess") is asserted
host-independently in tests/unit/test_worker_platform_unit.py, so it is covered on
any machine. What that test cannot do is exercise the part that genuinely needs
Windows: the one place that resolves the description into the real spawn kwargs
(platform_support.make_spawn_kwargs -- today a plain ``{}`` for both platforms, so
the child's console is left to the proactor) and actually spawns the child with that
here, on whatever host the suite runs on -- which is why
the module is gated on ``platform_support.IS_WINDOWS`` (so it is a clean skip on
the Linux dev/CI box and RUns on the ``windows-latest`` CI job, see
.github/workflows/run-unit-tests.yml).

The lifecycle itself uses the same stub model (OPENARC_WORKER_STUB=1) as
tests/unit/test_remote_vlm_worker_unit.py, so it needs no OpenVINO, GPU, or model
files -- it just proves the full spawn -> LOAD -> GENERATE -> FATAL+respawn ->
UNLOAD pipe works when driven by a Windows child. The assertion unique to Windows
is that the spawn added *no* console control, so the proactor left the child's
stdin/stdout/stderr valid and the worker reads its commands.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest  # type: ignore[import]

from src.engine.worker import platform_support as ps
from src.engine.worker import protocol as proto
from src.engine.worker.worker_client import RemoteOVGenAI_VLM
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType

pytestmark = pytest.mark.skipif(
    not ps.IS_WINDOWS,
    reason="exercises the real Windows spawn (hidden-console STARTUPINFO); "
    "skipped on a non-Windows host -- run it on the windows-latest CI job",
)


# -- the Windows spawn itself -----------------------------------------------------------------


def test_windows_spawn_adds_no_console_control() -> None:
    """The Windows-specific resolution 'build a hidden-console STARTUPINFO' became
    'add nothing': forcing a hidden console clobbered the proactor's
    STARTF_USESTDHANDLES wiring and left the child's stdin an invalid handle
    (OSError [WinError 6]: The handle is invalid) -- an invalid, SILENT worker that
    never sends LOAD_OK, so the 30s load budget expired. That was the integration
    failure this file proved was fixed: a clean spawn (no STARTUPINFO, no
    creationflags) leaves asyncio's proactor to wire the child's stdio, and the
    full lifecycle just below only runs if that wiring is valid."""
    extra = ps.make_spawn_kwargs(ps.current_platform())
    assert extra == {}, extra   # neither a STARTUPINFO nor creationflags


# -- the full lifecycle, driven by a Windows child ---------------------------------------------


def _load_config(tmp_path, name: str = "stub-vlm", model_type: ModelType = ModelType.VLM) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(tmp_path),
        model_name=name,
        model_type=model_type,
        engine=EngineType.OV_GENAI,
        device="CPU",
    )


def _gen_config(
    prompt: str, stream: bool = False, request_id: Optional[str] = None
) -> OVGenAI_GenConfig:
    return OVGenAI_GenConfig(prompt=prompt, stream=stream, request_id=request_id)


async def _wait_for_respawn(
    facade: RemoteOVGenAI_VLM, first_pid: Optional[int], timeout: float = 30.0
) -> None:
    async def _poll() -> None:
        while facade.worker_pid == first_pid or facade.status()["state"] != "ready":
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


async def _drain(facade: RemoteOVGenAI_VLM, gen_config: OVGenAI_GenConfig) -> list:
    run = facade.generate_type(gen_config)
    return [item async for item in run]


def test_stub_vlm_worker_full_lifecycle_on_windows(tmp_path, monkeypatch) -> None:
    """Full spawn -> LOAD -> GENERATE -> FATAL+respawn -> UNLOAD, driven by a real
    Windows child (the same stub used on POSIX). A new PID proves a fresh process
    respawned; the (console-control-free) spawn pinned by the test above is what
    left the child's stdin a valid handle so it could read at all. Passing this on
    Windows means
    the *whole* worker machine -- not just the decision -- works there."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path)

    async def _run() -> None:
        facade = RemoteOVGenAI_VLM(config, load_timeout=30.0)
        await facade.load_model(config)
        assert facade.status()["state"] == "ready"
        assert facade.worker_pid is not None

        items = await _drain(facade, _gen_config("hello world"))
        assert isinstance(items[0], dict)
        assert items[1] == "hello world"

        # A wedge (FATAL) respawns a fresh process on Windows too -- a different
        # PID is the positive proof, exactly as on POSIX.
        with pytest.raises(proto.RemoteWorkerError):
            await _drain(facade, _gen_config("FATAL please"))
        first_pid = facade.worker_pid
        await _wait_for_respawn(facade, first_pid)

        items = await _drain(facade, _gen_config("back up"))
        assert items[-1] == "back up"

        await facade._supervisor.unload()
        assert facade.status()["state"] == "closed"

    asyncio.run(_run())
