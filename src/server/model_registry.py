from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelStatus,
    ModelType,
)
from src.server.utils.config_hash import check_model_config_hash
from src.server.utils.context import resolve_context_window

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
    model_type: str = ""
    engine: str = ""
    device: str = ""
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    tool_call_parser: Optional[str] = None
    # Context window (tokens) advertised in /v1/models. Discovered from the model's
    # config.json at load time, or set explicitly via the load config.
    context_window: Optional[int] = None
    # The FULL resolved load configuration this record was created from (including
    # the resolved context_window). Kept on the record so the model can always be
    # re-registered (or its inference backend re-spawned) with identical parameters.
    load_config: Optional[ModelLoadConfig] = None
    max_tokens: Optional[int] = None  # Model-level default max_tokens (applied when a request omits it)


    def registered_models(self) -> dict:
        """Return only public fields as JSON-serializable dict."""
        result = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "engine": self.engine,
            "device": self.device,
            "runtime_config": self.runtime_config,
            "tool_call_parser": self.tool_call_parser,
            "context_window": self.context_window,
            "status": self.status.value,
            "time_loaded": self.time_loaded.isoformat(),
        }
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

    def add_on_loaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_loaded.append(callback)

    def add_on_unloaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_unloaded.append(callback)

    async def register_load(
        self,
        loader: ModelLoadConfig,
        force_recompile: bool = False,
    ) -> str:
        """Register and load a model, waiting for completion.

        Model names are unique across the registry: a second load with an
        existing name raises ValueError.

        Args:
            loader: The full load configuration. It is resolved (context window)
                and then stored verbatim on the record so the model can later be
                re-registered with identical parameters.
            force_recompile: When True, the compiled-model cache is invalidated
                and the pipeline is forced to recompile from the IR even if the
                configuration is UNCHANGED (so the config-hash gate would stay
                closed). This is driven by the --force-recompile / --fr flag on
                `openarc serve start` (server startup) and `openarc load`
                (POST /openarc/load) -- an explicit operator request for a fresh
                compile (e.g. after swapping the model files or host, or simply
                to be sure the pipeline was freshly built).

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

        # Resolve the effective context window. An explicit value on the load config
        # (e.g. `openarc add --context-window`, or the `context_window` key in
        # config.yaml) wins; otherwise it is discovered from the model's config.json.
        # Running it off the event loop keeps discovery (a small file read) from
        # stalling the loop.
        context_window = await asyncio.to_thread(
            resolve_context_window, loader.model_path, loader.context_window
        )

        # The inference engine only ever sees `loader` (via create_model_instance ->
        # OVGenAI_LLM/OVGenAI_VLM.load_model -> extract_scheduler_config_from_loader),
        # where the window becomes the compiled pipeline's MAX CONTENT WINDOW
        # (SchedulerConfig.max_num_batched_tokens, which openvino.genai uses to bound a
        # running sequence's KV-cache growth). Propagate the *resolved* value onto the
        # loader so a --context-window override actually bites at inference time, not
        # merely shows up in /v1/models. The record keeps the same resolved value so
        # what is advertised matches what is enforced.
        loader = loader.model_copy(update={"context_window": context_window})

        # Compiled-model invalidation gate: the OpenVINO cache holds blobs
        # compiled for THIS model's files + device, keyed without regard to the
        # runtime/scheduler configuration. If the config entry changed since the
        # last compile (a changed --runtime-config / --scheduler-config, a manual
        # edit of openarc_config.json, a re-run of `openarc add`, ...), the cached
        # blobs are stale and conflict with the new settings, so the cache is
        # cleared and the pipeline recompiles; the hash of the config being
        # compiled is persisted in the config file either way. Runs off the event
        # loop (config file I/O, and the cache directory can be large).
        #
        # A recompile is also FORCED even when the config is UNCHANGED (so the
        # config-hash gate would stay closed) by an operator request
        # (force_recompile=True): the --force-recompile / --fr flag on
        # `openarc serve start` / `openarc load`, which asks for a fresh compile
        # regardless (e.g. after the operator swapped the model files or host,
        # or simply to be sure the pipeline is freshly built).
        current_config_hash, config_changed = await asyncio.to_thread(
            check_model_config_hash, loader, force_recompile=force_recompile
        )
        if config_changed:
            logger.warning(
                f"[{loader.model_name}] Model config changed since the last compile; "
                f"compiled-model cache {loader.cache_dir!r} invalidated -- the model "
                f"will recompile with the new settings."
            )
        elif force_recompile:
            logger.warning(
                f"[{loader.model_name}] Forcing a recompile (--force-recompile / --fr); "
                f"the config is unchanged but the compiled-model cache {loader.cache_dir!r} "
                f"is invalidated so the pipeline rebuilds from the IR instead of re-importing "
                f"its existing (possibly stale) blobs."
            )
        # The record must carry the hash of the config it was compiled with so a
        # later ORDINARY load (server startup, POST /openarc/load) matches and
        # reuses the warm cache. An operator --force-recompile (force_recompile=True)
        # invalidates the cache but does NOT change the hash that is recorded/
        # persisted (the config is unchanged), so the next ordinary load still
        # finds a warm, matching cache.
        loader = loader.model_copy(update={"config_hash": current_config_hash})

        # Create a model record with LOADING status
        record = ModelRecord(
            model_path=loader.model_path,
            model_name=loader.model_name,
            model_type=loader.model_type,
            engine=loader.engine,
            device=loader.device,
            runtime_config=loader.runtime_config,
            tool_call_parser=(
                loader.tool_call_parser.value if loader.tool_call_parser else None
            ),
            context_window=context_window,
            max_tokens=loader.max_tokens,
            status=ModelStatus.LOADING,
            load_config=loader,
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

            # Start background unload task
            asyncio.create_task(self._unload_task(model_id))
            return True

    async def _load_task(self, model_id: str, load_config: ModelLoadConfig) -> None:
        """Background task to load a model and update its status."""
        try:
            # Load the model instance
            model_instance = await create_model_instance(load_config)

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

    async def status(self) -> dict:
        """Return registry status: total count and list of loaded models (public view)."""
        async with self._lock:
            models_public = [record.registered_models() for record in self._models.values()]
            return {
                "total_loaded_models": len(models_public),
                "models": models_public,
                "openai_model_names": [record.model_name for record in self._models.values()],
            }

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

    # Dynamic import and instantiation
    class_path = MODEL_CLASS_REGISTRY[key]
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Create and load model instance
    model_instance = model_class(load_config)
    await asyncio.to_thread(model_instance.load_model, load_config)
    return model_instance
