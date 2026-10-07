"""End-to-end test for the worker's per-session usage tracker.

Spawns a REAL worker child process whose model is a pure-Python stub that, like a
real pipeline, reports ``input_token`` in its metrics (no OpenVINO/GPU/model
files needed). It proves the actual worker wiring in ``worker_process._run_inference``
-- ``session_id`` -> ``_worker_sessions.augment`` -> the metrics carry
``session_context``/``session_delta``/``session_recalibrated`` back to the parent
-- and that it is safe to leave alone when no session id is present.

The key guarantee R3 depends on: brand-new process has an empty session map, so
the first time it sees a session it reports ``recalibrated=True`` (a re-base),
which is why, after a crash, goose's re-sent full conversation re-calibrates the
server-side session.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto
from src.engine.worker.supervisor import EOF, WorkerSupervisor
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_ENTRY = "openarc_worker_session_probe"

# The child's model: a pure-Python stub whose metrics carry a real (from the
# prompt) input_token so the worker's usage tracker has a number to track.
WORKER_MAIN_SRC = r'''
from __future__ import annotations

import asyncio
from typing import Any, Dict
from src.engine.worker.worker_process import _Worker, _configure_logging


def _input_token(gen_config: Any) -> int:
    # Deterministic from the prompt so different prompts => different context.
    prompt = gen_config.prompt or ""
    return max(1, len(str(prompt).split())) * 100


class _InputTokenStub:
    def __init__(self, load_config: Any) -> None:
        self.load_config = load_config

    def load_model(self, loader: Any) -> None:
        pass  # no pipeline to real build here

    async def generate_type(self, gen_config: Any) -> Any:
        it = _input_token(gen_config)
        base: Dict[str, Any] = {"input_token": it, "new_token": 1, "stream": bool(gen_config.stream)}
        if gen_config.stream:
            yield "chunk"
            yield base
        else:
            yield base
            yield "chunk"

    async def transcribe(self, gen_config: Any) -> Any:
        raise NotImplementedError

    async def cancel(self, request_id: str) -> bool:
        return False


class _ProbeWorker(_Worker):
    def _build_model(self, load_config: Any) -> Any:
        return _InputTokenStub(load_config)


def main() -> int:
    _configure_logging()
    worker = _ProbeWorker()
    try:
        asyncio.run(worker.run())
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1
'''


def _write_worker_module(tmp_path: Path) -> Path:
    module_dir = tmp_path / "worker"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / f"{WORKER_ENTRY}.py").write_text(WORKER_MAIN_SRC, encoding="utf-8")
    return module_dir


class _ProbeSupervisor(WorkerSupervisor):
    WORKER_ENTRY = WORKER_ENTRY

    def __init__(self, *args, module_dir: Optional[str] = None, extra_paths=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_dir = module_dir
        self._extra_paths = [p for p in (extra_paths or []) if p]

    def _build_env(self) -> dict:
        # Front the child's path with the temp module dir and the repo root so it
        # can import both the probe worker and ``src.*``.
        env = super()._build_env()
        paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        for path in [self._module_dir, *self._extra_paths]:
            if path and path not in paths:
                paths.insert(0, path)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env


def _load_config(model_dir: str) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=model_dir,
        model_name="probe-llm",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
    )


def _gen_config(prompt: str, session_id: Optional[str]) -> OVGenAI_GenConfig:
    return OVGenAI_GenConfig(prompt=prompt, stream=False, session_id=session_id)


def _expected_input_token(prompt: str) -> int:
    return max(1, len(prompt.split())) * 100


def _make_supervisor(model_dir: str, module_dir: Path) -> _ProbeSupervisor:
    return _ProbeSupervisor(
        "probe-llm",
        load_timeout=30.0,
        max_respawns=2,
        on_dead=None,
        module_dir=str(module_dir),
        extra_paths=[str(REPO_ROOT)],
    )


def _metrics_item(items: list) -> Optional[dict]:
    # The produced metrics dict is the one dictionary item in a drain.
    for item in items:
        if isinstance(item, dict) and "input_token" in item:
            return item
    return None


async def _drain(sup: _ProbeSupervisor, gen_config: OVGenAI_GenConfig) -> list:
    queue, result = await sup.begin_run(
        proto.OP_GENERATE, gen_config.model_dump_json(), gen_config.request_id
    )
    items: list = []
    while True:
        item = await queue.get()
        if item is EOF:
            break
        items.append(item)
    await result
    return items


def test_worker_reports_recalibrated_then_delta(tmp_path) -> None:
    """One process: first turn re-calibrates, a continued turn adds a delta. A
    brand-new process re-sees the session as recalibrated again (the re-base that
    happens after a crash), and a session-less request is left untouched."""
    module_dir = _write_worker_module(tmp_path)
    model_dir = str(tmp_path)

    async def _run():
        # Same process: turn 1 (recalibrate) then turn 2 (increment).
        sup = _make_supervisor(model_dir, module_dir)
        try:
            await sup.start(_load_config(model_dir))
            assert sup.status()["state"] == "ready"

            items1 = await _drain(sup, _gen_config("hi", "sess-1"))
            m1 = _metrics_item(items1)
            assert m1 is not None
            assert m1["session_id"] == "sess-1"
            assert m1["session_recalibrated"] is True
            assert m1["session_delta"] is None
            assert m1["session_context"] == _expected_input_token("hi")

            items2 = await _drain(sup, _gen_config("hello there friend how are you", "sess-1"))
            m2 = _metrics_item(items2)
            assert m2 is not None
            assert m2["session_recalibrated"] is False
            assert m2["session_context"] == _expected_input_token("hello there friend how are you")
            assert m2["session_delta"] == (
                _expected_input_token("hello there friend how are you")
                - _expected_input_token("hi")
            )
            assert m2["session_delta"] > 0
        finally:
            await sup.unload()

        # A fresh process (the crash/respawn analog) sees the SAME session id first
        # => recalibrated again => the server re-bases instead of adding.
        fresh = _make_supervisor(model_dir, module_dir)
        try:
            await fresh.start(_load_config(model_dir))
            assert fresh.status()["state"] == "ready"
            assert fresh.pid != sup.pid  # a genuinely fresh process
            mfresh = _metrics_item(await _drain(fresh, _gen_config("hi", "sess-1")))
            assert mfresh is not None
            assert mfresh["session_recalibrated"] is True
            assert mfresh["session_delta"] is None
        finally:
            await fresh.unload()

    asyncio.run(asyncio.wait_for(_run(), timeout=120))


def test_no_session_id_leaves_metrics_untouched(tmp_path) -> None:
    """With ``--sih`` off there is no session id on the request, so the worker
    must add none of the session_* keys even though the stub emits input_token."""
    module_dir = _write_worker_module(tmp_path)
    model_dir = str(tmp_path)

    async def _run():
        sup = _make_supervisor(model_dir, module_dir)
        try:
            await sup.start(_load_config(model_dir))
            items = await _drain(sup, _gen_config("hi", None))
        finally:
            await sup.unload()

        m = _metrics_item(items)
        assert m is not None
        assert "session_id" not in m
        assert "session_context" not in m
        assert "session_delta" not in m
        assert "session_recalibrated" not in m

    asyncio.run(asyncio.wait_for(_run(), timeout=120))
