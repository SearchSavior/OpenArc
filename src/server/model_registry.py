from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelStatus,
    ModelType,
    ToolCallParser,
)
from src.server.utils.context import (
    check_context_window_exceeded,
    resolve_context_window,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=False, slots=True)
class ModelRecord:
    # Private fields
    model_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    time_loaded: datetime = field(default_factory=datetime.utcnow)
    model_instance: Optional[Any] = field(default=None)  # Actual loaded model instance
    loading_task: Optional[asyncio.Task] = field(default=None)  # Background loading task
    status: ModelStatus = field(default=ModelStatus.LOADING)
    error_message: Optional[str] = field(default=None)  # Error message if loading failed

    # Public fields
    model_path: str = ""
    model_name: str = ""
    model_type: ModelType = ModelType.LLM
    engine: EngineType = EngineType.OV_GENAI
    device: str = ""
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    tool_call_parser: Optional[ToolCallParser] = None
    context_window: Optional[int] = None
    # Set when a pinned context_window exceeds the real max_position_embeddings.
    # Kept out of registered_models() so it never reaches /v1/models; the load
    # endpoint relays it to the CLI as a loud warning.
    context_window_warning: Optional[str] = None

    # Model-level request defaults surfaced from config.yaml. Stored as plain
    # dicts (already validated by ModelLoadConfig) and used to seed per-request
    # configs; an explicit request value always wins over these.
    # Model-level request defaults from config.yaml, keyed by block name
    # (e.g. 'sampler_config', 'kokoro_config'). Each value is a dict of only the
    # keys the author wrote and is merged under the per-request config.
    model_config_blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def block(self, name: str) -> Dict[str, Any]:
        """Return a named config block (e.g. 'kokoro_config'), or {} if unset."""
        return self.model_config_blocks.get(name) or {}

    def registered_models(self) -> dict:
        """Return only public fields as JSON-serializable dict."""
        result = {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "engine": self.engine.value,
            "device": self.device,
            "runtime_config": self.runtime_config,
            "tool_call_parser": (
                self.tool_call_parser.value if self.tool_call_parser else None
            ),
            "context_window": self.context_window,
            "status": self.status.value,
            "time_loaded": self.time_loaded.isoformat(),
        }
        if self.model_config_blocks:
            result["model_config_blocks"] = self.model_config_blocks
        if self.error_message:
            result["error_message"] = self.error_message
        return result

class ModelRegistry:
    """Tracks loaded models by private model_id. Async-safe."""

    def __init__(self):
        self._models: Dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()
        # Names of models that *should* be loaded for the server to be ready.
        # A model joins this set once it has successfully loaded and leaves it
        # only when an administrator explicitly unloads it. A model that drops
        # out of self._models for any other reason (e.g. an error-triggered
        # unload) stays here so readiness reports the server as not ready.
        self._expected_models: Set[str] = set()
        # Event subscribers
        self._on_loaded: List[Callable[[ModelRecord], Awaitable[None]]] = []
        self._on_unloaded: List[Callable[[ModelRecord], Awaitable[None]]] = []
        # Whether a full registry teardown (shut down on Ctrl-C) is running: makes
        # shutdown() idempotent and stops a late register_load or a re-entrant
        # register_unload (some facades call it inside their own unload_model) from
        # resurrecting a worker that is being torn down.
        self._shutting_down = False
        # Tasks started by register_unload; kept so shutdown() can await the live
        # ones. Pruned of completed tasks in register_unload so this stays bounded
        # over a long-running server.
        self._unload_tasks: List[asyncio.Task] = []
        # Per-model grace for a controlled shutdown; after it the subprocess
        # transport is force-closed, so a hung worker can't strand an open pipe on
        # a loop that is about to be closed.
        self._shutdown_timeout: float = 10.0

    def add_on_loaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_loaded.append(callback)

    def add_on_unloaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_unloaded.append(callback)

    async def register_load(self, loader: ModelLoadConfig) -> str:
        """Register and load a model, waiting for completion.

        Raises:
            ValueError: If model name already exists
            Exception: Any exception during loading is propagated to caller
        """
        # Check if model name already exists before loading
        async with self._lock:
            for existing_record in self._models.values():
                if existing_record.model_name == loader.model_name:
                    logger.info(f"Load failed! model_name '{loader.model_name}' already exists")
                    raise ValueError(f"model_name '{loader.model_name}' already registered")

        # Reject config blocks that don't match this model's model_type before
        # anything is loaded, so a mismatched config.yaml fails fast.
        loader.validate_config_blocks()

        # Opt-in advertisement: None stays None (nothing read); "auto" discovers
        # max_position_embeddings; a positive int is advertised as-is.
        context_window = await asyncio.to_thread(
            resolve_context_window, loader.model_path, loader.context_window
        )

        # A pinned int above the real max_position_embeddings advertises a window
        # the model can't use: log it loudly and stash it for the load endpoint to
        # relay to the CLI. None/ "auto" and an unknown real limit never warn.
        context_window_warning = await asyncio.to_thread(
            check_context_window_exceeded,
            loader.model_name,
            loader.model_path,
            loader.context_window,
        )
        if context_window_warning:
            logger.warning("Model '%s':\n%s", loader.model_name, context_window_warning)

        # Create a model record with LOADING status
        record = ModelRecord(
            model_path=loader.model_path,
            model_name=loader.model_name,
            model_type=loader.model_type,
            engine=loader.engine,
            device=loader.device,
            runtime_config=loader.runtime_config,
            tool_call_parser=loader.tool_call_parser,
            context_window=context_window,
            context_window_warning=context_window_warning,
            model_config_blocks=dict(loader.model_config_blocks or {}),
            status=ModelStatus.LOADING,
        )

        # Register the model record immediately
        async with self._lock:
            self._models[record.model_id] = record

        # Start loading task
        loading_task = asyncio.create_task(self._load_task(record.model_id, loader))

        # Update the record with the task reference
        async with self._lock:
            if record.model_id in self._models:
                self._models[record.model_id].loading_task = loading_task

        # Wait for loading to complete and propagate exceptions
        try:
            await loading_task
            # Check if loading succeeded
            async with self._lock:
                if record.model_id in self._models:
                    final_record = self._models[record.model_id]
                    if final_record.status == ModelStatus.FAILED:
                        error_msg = final_record.error_message or "Unknown error"
                        raise RuntimeError(f"Model loading failed: {error_msg}")
        except asyncio.CancelledError:
            raise RuntimeError("Model loading was cancelled")

        return record.model_id

    async def register_unload(self, model_name: str, administrative: bool = False) -> bool:
        """Unregister/unload a model by model_name. Returns True if found and unload task started.

        Args:
            model_name: Name of the model to unload.
            administrative: True when an operator explicitly requested the
                unload (e.g. via the API), as opposed to an internal
                error-triggered unload. An administrative unload removes the
                model from the readiness expectation set so it is no longer
                required for the server to be considered ready.
        """
        async with self._lock:
            # An explicit unload means the operator no longer wants this model
            # loaded, so stop requiring it for readiness. Do this regardless of
            # whether the model is still present, so an operator can clear a
            # model that already dropped out due to an earlier error.
            if administrative:
                self._expected_models.discard(model_name)

            # Find model_id by model_name
            model_id = None
            for mid, record in self._models.items():
                if record.model_name == model_name:
                    model_id = mid
                    break

            if model_id is None:
                return False

            # Start background unload task. Keep it so a full shutdown() can await
            # the live ones; prune completed ones so the list stays bounded.
            unload_task = asyncio.create_task(self._unload_task(model_id))
            self._unload_tasks = [t for t in self._unload_tasks if not t.done()] + [unload_task]
            return True

    async def _load_task(self, model_id: str, load_config: ModelLoadConfig) -> None:
        """Background task to load a model and update its status."""
        try:
            # Load the model instance
            model_instance = await create_model_instance(load_config)
            # Out-of-process engines (the remote VLM facade) report fatal
            # worker failures back to the registry that loaded them.
            if hasattr(model_instance, "_registry") and model_instance._registry is None:
                model_instance._registry = self

            # Update the record with successful loading
            async with self._lock:
                if model_id in self._models:
                    record = self._models[model_id]
                    record.model_instance = model_instance
                    record.status = ModelStatus.LOADED
                    record.loading_task = None
                    # The model is now serving, so it is expected to remain
                    # loaded for readiness purposes.
                    self._expected_models.add(record.model_name)
                else:
                    return

            # Fire loaded event callbacks outside the lock
            for cb in self._on_loaded:
                asyncio.create_task(cb(record))

        except Exception as e:
            # Log the full exception with traceback
            logger.error(f"Model loading failed for {load_config.model_name}", exc_info=True)

            # Update the record with failure status
            async with self._lock:
                if model_id in self._models:
                    record = self._models[model_id]
                    record.status = ModelStatus.FAILED
                    record.error_message = str(e)
                    record.loading_task = None

    async def _unload_task(self, model_id: str) -> None:
        """Background task to unload a model and clean up resources."""
        try:
            async with self._lock:
                if model_id not in self._models:
                    return
                record = self._models[model_id]
                model_instance = record.model_instance

            # Call the model's unload_model method if it exists and model is loaded
            if model_instance and hasattr(model_instance, 'unload_model'):
                unload_fn = getattr(model_instance, 'unload_model')
                try:
                    # Prefer (registry, model_name) signature used by OVGenAI_* classes
                    result = unload_fn(self, record.model_name)
                except TypeError:
                    # Fallback to no-arg sync unload (e.g., Whisper)
                    result = unload_fn()
                # Await if coroutine/awaitable
                if inspect.isawaitable(result):
                    await result

            # Remove from registry
            async with self._lock:
                removed_record = None
                if model_id in self._models:
                    record = self._models[model_id]
                    # Cancel loading task if still running
                    if record.loading_task and not record.loading_task.done():
                        record.loading_task.cancel()
                    removed_record = self._models.pop(model_id)
                else:
                    removed_record = None
            if removed_record is not None:
                for cb in self._on_unloaded:
                    asyncio.create_task(cb(removed_record))

        except Exception as e:
            logger.info(f"Error during model unload: {e}")

    async def _shutdown_one(self, record: "ModelRecord") -> None:
        """Tear down one model: run its ``unload_model`` (which closes the
        supervised worker's subprocess transport *while the loop is alive*) and
        then unconditionally release that transport, so a hung or failed unload
        can never leave an open pipe for GC to close after ``loop.close()`` --
        the ``BaseSubprocessTransport.__del__`` -> ``Event loop is closed`` noise.
        In-process engines have no ``_supervisor`` (a clean no-op here).
        """
        instance = record.model_instance
        if instance is None:
            return
        # Best-effort graceful ask the worker to exit, then wait for it.
        unload_fn = getattr(instance, "unload_model", None)
        if unload_fn is not None:
            try:
                result = unload_fn(self, record.model_name)
            except TypeError:
                try:
                    result = unload_fn()
                except TypeError:
                    result = None
            if inspect.isawaitable(result):
                try:
                    await asyncio.wait_for(result, self._shutdown_timeout)
                except asyncio.CancelledError:
                    raise  # a real cancellation must propagate
                except Exception as e:
                    logger.warning(
                        f"[{record.model_name}] worker unload unfinished/failed "
                        f"({e!r}); forcing transport close"
                    )
        # Whatever the outcome, force the subprocess transport closed and drain its
        # pipe-reader tasks. Both are idempotent (see supervisor) -- this release
        # is the actual step that keeps Ctrl-C quiet.
        supervisor = getattr(instance, "_supervisor", None)
        if supervisor is not None:
            try:
                await supervisor._await_pipe_tasks()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            try:
                supervisor._close_process_streams()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def shutdown(self) -> None:
        """Tear down every registered worker on process exit.

        Must run on the *live* event loop (the ASGI ``lifespan`` shutdown half,
        the code after ``yield`` in server/main.py). It is the single reason a
        controlled Ctrl-C stop is quiet rather than spamming::

            Exception ignored in: <function BaseSubprocessTransport.__del__ ...>
              ... base_subprocess.py (close) -> unix_events (write_eof) ->
                  base_events (call_soon -> _check_closed)
            RuntimeError: Event loop is closed

        That traceback is Python's GC reaping a *still-open* subprocess pipe
        transport after the loop has already been closed: on Ctrl-C nothing had
        closed the transport (``supervisor.unload`` was never reached), so the
        ``__del__`` does the closing -- too late. The fix is to close every
        transport here, while the loop still runs. Closing a transport is
        platform-agnostic, so the same call covers both POSIX
        (UnixReadPipeTransport.write_eof) and Windows (the proactor / the child's
        stdin write handle) -- which is exactly why the reported traceback is
        identical on linux and windows.

        Idempotent: a second call is a no-op (``self._shutting_down``).
        """
        if self._shutting_down:
            return

        # 1) Take ownership of everything and stop further mutation, under the lock,
        #    so neither a late register_load nor a re-entrant register_unload can
        #    spawn a fresh unload task that GC the loop after we've gone.
        async with self._lock:
            self._shutting_down = True
            snapshot = list(self._models.items())
            self._models.clear()
            self._expected_models.clear()
            drain: List[asyncio.Task] = [t for t in self._unload_tasks if not t.done()]
            self._unload_tasks = []
            for _mid, record in snapshot:
                t = record.loading_task
                if t is not None and not t.done():
                    t.cancel()
                    drain.append(t)
        # 2) Let the captured background work finish, so the loop holds no callbacks
        #    that still reference these record objects.
        for fut in drain:
            try:
                await fut
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        # 3) Tear each model down. ``_shutdown_one`` re-reads record.model_instance,
        #    so a load that completed just before the cancel still has its supervisor
        #    (subprocess + transport) closed here.
        for _mid, record in snapshot:
            try:
                await self._shutdown_one(record)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error shutting down model {record.model_name}: {e}")
        # 4) Fire on_unloaded so the WorkerRegistry cancels its per-model queue
        #    worker tasks; reusing _on_model_unloaded keeps one queue-cancel path
        #    rather than growing a second one.
        for cb in self._on_unloaded:
            for _mid, record in snapshot:
                try:
                    await cb(record)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"on_unloaded callback failed during shutdown: {e}")

    async def status(self) -> dict:
        """Return registry status: total count and list of loaded models (public view)."""
        async with self._lock:
            models_public = [record.registered_models() for record in self._models.values()]
            return {
                "total_loaded_models": len(models_public),
                "models": models_public,
                "openai_model_names": [record.model_name for record in self._models.values()],
            }

    def get_context_window_warning(self, model_name: str) -> Optional[str]:
        """The over-sized-context warning stashed on a model's record, or None.

        None for an unset context_window, for "auto", or if the model is gone.
        Consulted by POST /openarc/load so the CLI can print it.
        """
        for record in self._models.values():
            if record.model_name == model_name:
                return record.context_window_warning
        return None

    async def readiness(self) -> dict:
        """Return readiness: ready only when every expected model is loaded.

        A model is "expected" once it has successfully loaded and until it is
        administratively unloaded. The server is ready when there is at least
        one expected model and all expected models currently have a LOADED
        record. Any expected model that is missing or not yet LOADED (e.g.
        unloaded due to an error, or still loading) makes the server not ready,
        as does having no models expected at all.
        """
        async with self._lock:
            loaded = {
                record.model_name
                for record in self._models.values()
                if record.status == ModelStatus.LOADED
            }
            expected = set(self._expected_models)
            missing = sorted(expected - loaded)
            return {
                "ready": bool(expected) and not missing,
                "expected_models": sorted(expected),
                "missing_models": missing,
            }

# Registry mapping (engine, model_type) to model class paths
MODEL_CLASS_REGISTRY = {
    (EngineType.OV_GENAI, ModelType.LLM): "src.engine.ov_genai.llm.OVGenAI_LLM",
    (EngineType.OV_GENAI, ModelType.VLM): "src.engine.ov_genai.vlm.OVGenAI_VLM",
    (EngineType.OV_GENAI, ModelType.WHISPER): "src.engine.ov_genai.whisper.OVGenAI_Whisper",
    (EngineType.OPENVINO, ModelType.QWEN3_ASR): "src.engine.openvino.qwen3_asr.qwen3_asr.OVQwen3ASR",
    (EngineType.OPENVINO, ModelType.KOKORO): "src.engine.openvino.kokoro.OV_Kokoro",
    (EngineType.OPENVINO, ModelType.QWEN3_TTS_CUSTOM_VOICE): "src.engine.openvino.qwen3_tts.qwen3_tts.OVQwen3TTS",
    (EngineType.OPENVINO, ModelType.QWEN3_TTS_VOICE_DESIGN): "src.engine.openvino.qwen3_tts.qwen3_tts.OVQwen3TTS",
    (EngineType.OPENVINO, ModelType.QWEN3_TTS_VOICE_CLONE): "src.engine.openvino.qwen3_tts.qwen3_tts.OVQwen3TTS",
    (EngineType.OV_OPTIMUM, ModelType.EMB): "src.engine.optimum.optimum_emb.Optimum_EMB",
    (EngineType.OV_OPTIMUM, ModelType.RERANK): "src.engine.optimum.optimum_rr.Optimum_RR",
}

async def create_model_instance(load_config: ModelLoadConfig) -> Any:
    """Factory function to create the appropriate model instance based on engine type."""
    key = (load_config.engine, load_config.model_type)

    if key not in MODEL_CLASS_REGISTRY:
        available = [f"{engine.value}/{model.value}" for engine, model in MODEL_CLASS_REGISTRY.keys()]
        error_msg = (
            f"Combination '{load_config.engine.value}/{load_config.model_type.value}' "
            f"not supported. Available: {', '.join(available)}"
        )
        logger.info(f"Model load failed: {error_msg}")
        raise ValueError(error_msg)

    # Dynamic import of the engine class.
    class_path = MODEL_CLASS_REGISTRY[key]
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Lazy imports: src.engine's package __init__ imports the engine classes,
    # and those import back into this module, so they must not be imported at
    # module level here (circular import).
    from src.engine.worker.worker_client import (
        RemoteOVGenAI_LLM,
        RemoteOVGenAI_VLM,
        RemoteOVGenAI_Whisper,
    )
    from src.engine.worker.plain.worker_client import (
        RemoteOV_Kokoro,
        RemoteOVQwen3ASR,
        RemoteOVQwen3TTS,
    )

    # OpenVINO models run in a dedicated worker process instead of in the
    # server process:
    #   * GenAI (VLM/LLM/Whisper): pipelines share a process-wide singleton
    #     ov::Core, and a wedged GPU plugin poisons it for the life of the
    #     process -- no in-process unload/reload can ever fix that.
    #   * plain OpenVINO (Kokoro/Qwen3-ASR/Qwen3-TTS): for SEGFAULT
    #     ISOLATION -- a native crash in inference takes down only the
    #     worker process, which is respawned with a fresh pipeline.
    # The facade owns a supervised subprocess, so unload = terminate
    # (guaranteed clean), reload = fresh process, and a wedged/crashed worker
    # is respawned transparently within a per-load budget.
    # OPENARC_OVGENAI_WORKER=0 (or OPENARC_VLM_WORKER=0 for VLMs) and
    # OPENARC_OPENVINO_WORKER=0 restore the historical in-process behaviour.
    #
    # The facade is chosen by (engine, model_type) -- the same key as
    # MODEL_CLASS_REGISTRY -- BEFORE instantiating the real engine: several
    # constructors have side effects (reading model files, creating ov.Core,
    # allocating tensors), which must not run in the server process when the
    # model is about to load in a worker.
    _WORKER_FACADES = {
        (EngineType.OV_GENAI, ModelType.VLM): RemoteOVGenAI_VLM,
        (EngineType.OV_GENAI, ModelType.LLM): RemoteOVGenAI_LLM,
        (EngineType.OV_GENAI, ModelType.WHISPER): RemoteOVGenAI_Whisper,
        (EngineType.OPENVINO, ModelType.KOKORO): RemoteOV_Kokoro,
        (EngineType.OPENVINO, ModelType.QWEN3_ASR): RemoteOVQwen3ASR,
        (EngineType.OPENVINO, ModelType.QWEN3_TTS_CUSTOM_VOICE): RemoteOVQwen3TTS,
        (EngineType.OPENVINO, ModelType.QWEN3_TTS_VOICE_DESIGN): RemoteOVQwen3TTS,
        (EngineType.OPENVINO, ModelType.QWEN3_TTS_VOICE_CLONE): RemoteOVQwen3TTS,
    }
    facade_cls = _WORKER_FACADES.get((load_config.engine, load_config.model_type))
    if facade_cls is not None and _worker_enabled(
        load_config.engine, load_config.model_type
    ):
        model_instance = facade_cls(load_config)
    else:
        model_instance = model_class(load_config)

    # Load the model instance: remote facades load asynchronously (they spawn
    # a subprocess); in-process engines keep their blocking load off the
    # event loop.
    load_fn = model_instance.load_model
    if inspect.iscoroutinefunction(load_fn):
        await load_fn(load_config)
    else:
        await asyncio.to_thread(load_fn, load_config)
    return model_instance


def _worker_enabled(engine: EngineType, model_type: ModelType) -> bool:
    """Whether a model of this engine/type runs in a worker process (stage 1-3).

    Optimum models (emb/rerank) still run in-process (a later stage).
    """
    if engine == EngineType.OV_GENAI:
        return _ovgenai_worker_enabled(model_type)
    if engine == EngineType.OPENVINO:
        return _openvino_worker_enabled(model_type)
    return False


def _ovgenai_worker_enabled(model_type: ModelType) -> bool:
    """Whether an OpenVINO GenAI model of this type runs in a worker process.

    OPENARC_OVGENAI_WORKER is the master switch (default on);
    OPENARC_VLM_WORKER additionally gates VLMs (stage-1 escape hatch).
    """
    if os.getenv("OPENARC_OVGENAI_WORKER", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    if model_type == ModelType.VLM:
        return os.getenv("OPENARC_VLM_WORKER", "1").strip().lower() not in ("0", "false", "off", "no")
    return True


def _openvino_worker_enabled(model_type: ModelType) -> bool:
    """Whether a plain-OpenVINO model (Kokoro/Qwen3-ASR/Qwen3-TTS) of this
    type runs in a worker process (stage 3, for segfault isolation).

    OPENARC_OPENVINO_WORKER is the master switch (default on); set it to 0 to
    restore the historical in-process behaviour.
    """
    return (
        os.getenv("OPENARC_OPENVINO_WORKER", "1").strip().lower()
        not in ("0", "false", "off", "no")
    )
