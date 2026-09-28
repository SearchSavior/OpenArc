"""
Integration test: an OpenArc inference worker is RESTARTED when it reports an
unrecoverable OpenVINO error (a wedged device / CL_OUT_OF_RESOURCES), using a
REAL ``openvino_genai`` pipeline running on the CPU device.

The unit tests (``tests/unit/test_*_worker_unit.py``) exercise the whole
supervisor/respawn machinery with a *stubbed* model (``OPENARC_WORKER_STUB=1``
- no OpenVINO, GPU, or model files). This test instead runs a genuine one:
it (re)generates a tiny ``openvino_genai`` LLM
(see ``tests/integration/_generate_tiny_ovllm.py``) and loads it in a real
worker subprocess on device ``CPU``. It then makes a request that drives the
worker's real error ladder - a ``CL_OUT_OF_RESOURCES``-style failure - so the
real worker reports ``MSG_FATAL`` and exits, and we assert the real supervisor
spawns a fresh process (a new PID, back to ``READY``) instead of leaving the
model dead.

The unrecoverable condition only really happens on a wedged GPU/driver, not on
CPU, so the worker we spawn subclasses the *real* worker loop
(``src.engine.worker.worker_process._Worker``) and overrides only which model
object it builds. The model is a wrapper around the REAL ``OVGenAI_LLM`` (real
pipeline, real load on CPU, genuine inference for a healthy request); a request
carrying a control marker raises a ``CL_OUT_OF_RESOURCES`` ``RuntimeError`` -
exactly the exception class the worker's ``is_non_recoverable_error`` tells
apart from an ordinary per-request error. So the FATAL + exit + respawn path
exercised here is the entire production code; only the *model* is a test double
around it - and no production file is touched.

The small model is committed under ``tests/integration/fixtures/`` (4 MB; well
under the 10 MB threshold for committing a fixture). If it is missing on a fresh
checkout, the test regenerates it on the fly by running the generator script
with the ``optimum-intel`` venv; if that is not possible it skips.

Run with:

    pytest -W ignore tests/integration/test_worker_respawn_integration.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto
from src.engine.worker.supervisor import EOF, WorkerSupervisor
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MODEL = REPO_ROOT / "tests" / "integration" / "fixtures" / "tiny-gpt2-ov"
GENERATOR = REPO_ROOT / "tests" / "integration" / "_generate_tiny_ovllm.py"
# optimum-intel (the HF -> OpenVINO converter) lives in its own venv (see
# setup.sh) separate from the openarc-devel venv that runs the tests.
OPTIMUM_INTEL_PY = os.environ.get(
    "OPTIMUM_INTEL_PYTHON",
    "/home/eckhard/.local/pyvenv/optimum-intel/bin/python",
)

# The control token in a request's prompt that makes the worker raise a
# non-recoverable error. Sent verbatim; the worker reads it from env so the test
# never has to (they share this default).
WEDGE_MARKER = "OV_WEDGE"

# The worker module the test spawns as the child's entry point. It subclasses
# the real worker loop and only swaps the model object it builds for one that
# wraps the REAL OVGenAI_LLM but raises a non-recoverable error for a request
# carrying the wedge marker. Everything else (the protocol loop, the real
# is_non_recoverable_error classification, the FATAL + os._exit, the supervisor
# respawn) is the genuine production code. It is written to a temp dir and
# placed at the front of the child's PYTHONPATH, so no production module changes.
WORKER_MAIN_SRC = r'''
from __future__ import annotations

import asyncio
import os
import sys

from src.engine.worker import protocol as proto
from src.engine.worker.worker_process import _Worker, _configure_logging
from src.engine.ov_genai.llm import OVGenAI_LLM

WEDGE_MARKER = os.environ.get("OPENARC_WORKER_WEDGE_MARKER", "OV_WEDGE")


def _payload_text(gen_config: Any) -> str:
    """Read the control text out of a (validated) OVGenAI_GenConfig, the same
    way the production stub worker does, so a prompt carrying the marker is
    detected without any extra wire fields."""
    text = gen_config.prompt or ""
    for msg in gen_config.messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            text += " " + content
    return text


class _WedgeLLM:
    """Wraps a real OVGenAI_LLM (real pipeline, real load on CPU). A request
    carrying the wedge marker raises a CL_OUT_OF_RESOURCES-style error so the
    worker's *real* error ladder takes the process down; any other request runs
    genuine inference and returns real (metrics, text) output."""

    def __init__(self, load_config: Any) -> None:
        self.load_config = load_config
        self._inner = OVGenAI_LLM(load_config)

    def load_model(self, loader: Any) -> None:
        # REAL openvino_genai.LLMPipeline build on the loader's device (CPU).
        self._inner.load_model(loader)

    async def generate_type(self, gen_config: Any) -> Any:
        if WEDGE_MARKER and WEDGE_MARKER in _payload_text(gen_config):
            # This is what a wedged GPU/driver surfaces through the openvino C++
            # layer; proto.is_non_recoverable_error classifies it, the worker
            # reports FATAL and exits, and the supervisor respawns a fresh one.
            raise RuntimeError(
                "CL_OUT_OF_RESOURCES executing the LLM pipeline "
                "(injected: a transient, non-recoverable device failure)"
            )
        # Healthy path: genuine inference. This openvino-genai build's
        # pipeline.get_tokenizer() carries no detokenizer for this model, so
        # decode with the AutoTokenizer the real OVGenAI_LLM already loads.
        import openvino as ov

        prompt = gen_config.prompt or "ping"
        ids = self._inner.encoder_tokenizer.encode(prompt, return_tensors="np")
        cfg = self._inner.model.get_generation_config()
        cfg.max_new_tokens = 8
        cfg.temperature = 0.0
        cfg.top_k = 1
        cfg.top_p = 1.0
        result = await asyncio.to_thread(self._inner.model.generate, ov.Tensor(ids), cfg)
        decoded = self._inner.encoder_tokenizer.decode(
            list(result.tokens), skip_special_tokens=True
        )
        text = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
        yield {
            "new_token": len(result.tokens),
            "stream": False,
            "load_time (s)": 0.0,
        }
        yield text if text is not None else ""

    async def transcribe(self, gen_config: Any) -> Any:
        raise NotImplementedError("the wedge test model serves LLM only")


class _WedgeWorker(_Worker):
    def _build_model(self, load_config: Any) -> Any:
        # The real workloop, the real protocol, the real error ladder; only the
        # model object being built differs (a test double around the real one).
        return _WedgeLLM(load_config)


def main() -> int:
    _configure_logging()
    worker = _WedgeWorker()
    try:
        asyncio.run(worker.run())
        return 0
    except SystemExit as e:  # the process is the unit under test
        return int(e.code) if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        # Mirror the real worker's top-level guard: anything uncaught here means
        # the pipeline's ov::Core is poisoned, so it is FATAL and the process
        # must exit so the parent can spawn a clean one.
        sys.stdout.buffer.write(
            proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e))
        )
        sys.stdout.buffer.flush()
        return 1
'''


class _FakeRegistry:
    """Minimal stand-in for ModelRegistry for tests that only observe how the
    supervisor's on_dead callback reports a permanently dead (quarantined)
    worker to the registry."""

    def __init__(self) -> None:
        self.unloaded: list = []

    async def register_unload(self, model_name: str, administrative: bool = False) -> bool:
        self.unloaded.append(model_name)
        return True


class _WedgeWorkerSupervisor(WorkerSupervisor):
    """WorkerSupervisor that runs the test's worker module instead of the real
    one; it injects the temp dir holding that module into the child's path so
    the ``from <module> import main`` entry point resolves, and it keeps the
    test-side repo root on the path so the child can import ``src.*``."""

    WORKER_ENTRY = "openarc_worker_respawn_wedge_main"

    def __init__(
        self,
        *args: Any,
        module_dir: Optional[str] = None,
        extra_paths: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._module_dir = module_dir
        self._extra_paths = [p for p in (extra_paths or []) if p]

    def _build_env(self) -> dict:
        env = super()._build_env()
        paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        for path in [self._module_dir, *self._extra_paths]:
            if path and path not in paths:
                paths.insert(0, path)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env


def _ensure_model() -> Optional[Path]:
    """Return the model dir, generating it on the fly if it is not committed.

    Prefers the committed fixture; if it is absent (fresh checkout), runs the
    generator script once; skips (returns None) if generation is not possible
    here (no network / no optimum-intel venv) rather than failing red.
    """
    if FIXTURE_MODEL.is_dir() and (FIXTURE_MODEL / "openvino_model.xml").exists():
        return FIXTURE_MODEL
    if not GENERATOR.is_file():
        return None
    if not os.path.exists(OPTIMUM_INTEL_PY):
        return None
    # Cache the HF download / convert staging OUTSIDE the repo (the committed
    # model is just the small IR+tokenizer dir); a repo-internal cache would
    # bloat the tree and tempt a stray commit.
    cache_dir = os.path.join(tempfile.gettempdir(), "openarc-test-models")
    env = dict(os.environ, OPENARC_TEST_MODEL_CACHE=cache_dir)
    try:
        # Run the generator with the optimum-intel venv (optimum/optimum-intel
        # live there, not in the openarc-devel venv running the tests).
        returncode = os.system(
            f"{OPTIMUM_INTEL_PY!r} {str(GENERATOR)!r}"
        )
        ok = returncode == 0 if isinstance(returncode, int) else True
    except Exception:
        ok = False
    if ok and FIXTURE_MODEL.is_dir() and (FIXTURE_MODEL / "openvino_model.xml").exists():
        return FIXTURE_MODEL
    return None


def _write_worker_module(tmp_path: Path) -> Path:
    module_dir = tmp_path / "worker"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / f"{_WedgeWorkerSupervisor.WORKER_ENTRY}.py").write_text(
        WORKER_MAIN_SRC, encoding="utf-8"
    )
    return module_dir


def _load_config(
    model_dir: Path,
    model_name: str,
    worker_max_respawns: Optional[int],
) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(model_dir),
        model_name=model_name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        worker_max_respawns=worker_max_respawns,
    )


def _gen_config(prompt: str, stream: bool = False) -> OVGenAI_GenConfig:
    return OVGenAI_GenConfig(
        prompt=prompt,
        stream=stream,
        max_tokens=8,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )


async def _drain(sup: WorkerSupervisor, gen_config: OVGenAI_GenConfig) -> list:
    """Send one request and collect its items; re-raise the result future's
    exception (so an unrecoverable error surfaces as RemoteWorkerDeadError)."""
    queue, result = await sup.begin_run(
        proto.OP_GENERATE, gen_config.model_dump_json(), gen_config.request_id
    )
    items: list = []
    while True:
        item = await queue.get()
        if item is EOF:
            break
        items.append(item)
    await result  # raises if the worker reported an error/fatal
    return items


async def _wait_for_respawn(
    sup: WorkerSupervisor, first_pid: Optional[int], timeout: float = 120.0
) -> None:
    """Wait until the supervisor has detected the death and the respawned worker
    is ready again (a fresh PID proves a fresh process / fresh ov::Core)."""

    async def _poll() -> None:
        while sup.pid == first_pid or sup.status()["state"] != "ready":
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


async def _wait_for_state(
    sup: WorkerSupervisor, state: str, timeout: float = 120.0
) -> None:

    async def _poll() -> None:
        while sup.status()["state"] != state:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


def _make_supervisor(
    load_config: ModelLoadConfig,
    registry: _FakeRegistry,
    module_dir: Path,
) -> _WedgeWorkerSupervisor:
    async def on_dead() -> None:
        await registry.register_unload(load_config.model_name)

    return _WedgeWorkerSupervisor(
        load_config.model_name,
        load_timeout=120.0,
        max_respawns=2,
        on_dead=on_dead,
        module_dir=str(module_dir),
        extra_paths=[str(REPO_ROOT)],
    )


def test_worker_is_respawned_after_unrecoverable_openvino_error(tmp_path: Any) -> None:
    """An unrecoverable (CL_OUT_OF_RESOURCES-style) error while serving: the
    worker reports it and exits, and the supervisor spawns a fresh process that
    comes back up READY. With the default budget (2) a single crash is within
    budget, so the model is reloaded and is NOT quarantined; the fresh worker
    then serves a normal request again (proof the respawn yields a usable
    process, not just another one that also dies)."""
    model_dir = _ensure_model()
    if model_dir is None:
        pytest.skip(
            f"tiny OVGenAI model fixture missing at {FIXTURE_MODEL}; generate it "
            f"with `{Path(OPTIMUM_INTEL_PY)} {GENERATOR}` (set OPTIMUM_INTEL_PYTHON "
            f"to the optimum-intel venv, if different) or commit it."
        )

    async def _run() -> None:
        registry = _FakeRegistry()
        module_dir = _write_worker_module(tmp_path)
        load_config = _load_config(model_dir, "wedge-llm", worker_max_respawns=None)
        sup = _make_supervisor(load_config, registry, module_dir)
        try:
            await sup.start(load_config)
            assert sup.status()["state"] == "ready"
            pid0 = sup.pid
            assert pid0 is not None
            # None in the load config => the supervisor's own default budget (2).
            assert sup.status()["max_respawns"] == 2

            # (1) A normal request on the real pipeline: it serves real output
            #     (metrics dict, then text), and the worker stays up.
            healthy = await _drain(sup, _gen_config("ping please"))
            assert healthy[0].get("stream") is False
            assert isinstance(healthy[-1], str)
            assert sup.pid == pid0
            assert sup.status()["state"] == "ready"
            assert registry.unloaded == []

            # (2) An unrecoverable error fires the worker's real FATAL + exit,
            #     and the supervisor respawns a fresh process within budget.
            with pytest.raises(proto.RemoteWorkerDeadError) as exc_info:
                await _drain(sup, _gen_config(f"{WEDGE_MARKER} say hi"))
            # The death is classified as a non-recoverable device failure.
            assert "CL_OUT_OF_RESOURCES" in str(exc_info.value)

            await _wait_for_respawn(sup, pid0)
            assert sup.pid != pid0  # a genuinely fresh process / Core
            assert sup.status()["state"] == "ready"
            assert sup.status()["respawns"] == 1
            # Within the default budget of 2, so the model is reloaded, not quarantined.
            assert registry.unloaded == []

            # (3) The respawned worker is usable again: a normal request succeeds.
            again = await _drain(sup, _gen_config("ping again"))
            assert again[0].get("stream") is False
            assert isinstance(again[-1], str)
            assert registry.unloaded == []
        finally:
            await sup.unload()

    asyncio.run(asyncio.wait_for(_run(), timeout=600))


def test_worker_max_respawns_zero_never_quarantines(tmp_path: Any) -> None:
    """worker_max_respawns=0 means NO limit, on a REAL worker: it is reloaded on
    every unrecoverable crash (here 3) and is never unloaded/quarantined."""
    model_dir = _ensure_model()
    if model_dir is None:
        pytest.skip(
            f"tiny OVGenAI model fixture missing at {FIXTURE_MODEL}; generate it "
            f"with `{Path(OPTIMUM_INTEL_PY)} {GENERATOR}` or commit it."
        )

    async def _run() -> None:
        registry = _FakeRegistry()
        module_dir = _write_worker_module(tmp_path)
        load_config = _load_config(model_dir, "wedge-llm", worker_max_respawns=0)
        sup = _make_supervisor(load_config, registry, module_dir)
        try:
            await sup.start(load_config)
            assert sup.status()["max_respawns"] == 0  # override applied in start()

            crashes = 3
            for _ in range(crashes):
                pid = sup.pid
                assert pid is not None
                with pytest.raises(proto.RemoteWorkerDeadError):
                    await _drain(sup, _gen_config(f"{WEDGE_MARKER} crash"))
                await _wait_for_respawn(sup, pid)
                assert sup.pid != pid  # each crash respawns a fresh process
                assert sup.status()["state"] == "ready"
                assert registry.unloaded == []  # 0 => never quarantined
            assert sup.status()["respawns"] == crashes
        finally:
            await sup.unload()

    asyncio.run(asyncio.wait_for(_run(), timeout=600))


def test_worker_max_respawns_budget_then_quarantines(tmp_path: Any) -> None:
    """The configurable budget, end-to-end with a real worker: with
    worker_max_respawns=1 the worker is reloaded ONCE after an unrecoverable
    crash, then quarantined (unloaded from the registry) on the next one - the
    original "reloaded twice under the default of 2, quit on the 3rd" behaviour,
    now parameterised down to a single respawn."""
    model_dir = _ensure_model()
    if model_dir is None:
        pytest.skip(
            f"tiny OVGenAI model fixture missing at {FIXTURE_MODEL}; generate it "
            f"with `{Path(OPTIMUM_INTEL_PY)} {GENERATOR}` or commit it."
        )

    async def _run() -> None:
        registry = _FakeRegistry()
        module_dir = _write_worker_module(tmp_path)
        load_config = _load_config(model_dir, "wedge-llm", worker_max_respawns=1)
        sup = _make_supervisor(load_config, registry, module_dir)
        try:
            await sup.start(load_config)
            assert sup.status()["max_respawns"] == 1  # override applied in start()

            # First crash: within the budget of 1 -> respawn (not quarantined).
            first_pid = sup.pid
            assert first_pid is not None
            with pytest.raises(proto.RemoteWorkerDeadError):
                await _drain(sup, _gen_config(f"{WEDGE_MARKER} first"))
            await _wait_for_respawn(sup, first_pid)
            assert sup.status()["respawns"] == 1
            assert registry.unloaded == []  # budget not yet spent

            # Second crash: the budget is spent -> the model is quarantined.
            second_pid = sup.pid
            with pytest.raises(proto.RemoteWorkerDeadError):
                await _drain(sup, _gen_config(f"{WEDGE_MARKER} second"))
            await _wait_for_state(sup, "dead")
            for _ in range(200):
                if registry.unloaded:
                    break
                await asyncio.sleep(0.01)
            assert registry.unloaded == [load_config.model_name]
            assert sup.pid == second_pid  # no third respawn
        finally:
            await sup.unload()

    asyncio.run(asyncio.wait_for(_run(), timeout=600))
