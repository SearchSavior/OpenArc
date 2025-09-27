from __future__ import annotations

import asyncio
import uuid
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Callable, Awaitable, List

from pydantic import BaseModel, Field

class ModelLoadConfig(BaseModel):
    model_path: str = Field(
        description="""
        Top level path to directory containing OpenVINO IR converted model.
        
        OpenArc does not support runtime conversion and cannot pull from HF.""")
    model_name: str = Field(
        ...,
        description="""
        - Public facing name of the loaded model attached to a private model_id
        - Calling /v1/models will report model names from this list
        - model_name is decoupled from last segment of model_path, though in practice you should use that value.
        """
        )
    model_type: TaskType = Field(...)
    engine: EngineType = Field(...)
    device: str = Field(
        ...,
        description="""
        Device used to load the model.
        """
        )
    runtime_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional OpenVINO runtime properties.")

class ModelUnloadConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to unload")

class ModelStatus(str, Enum):
    """Model loading status.
    
    Options:
    - LOADING: Model is currently being loaded in the background
    - LOADED: Model has been successfully loaded and is ready for inference
    - FAILED: Model loading failed
    """
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"

class TaskType(str, Enum):
    """Internal routing to the correct inference pipeline.

    Options:
    - text_to_text: Text-to-text LLM models
    - image_to_text: Image-to-text VLM models"""
    
    TEXT_TO_TEXT = "text_to_text"
    IMAGE_TO_TEXT = "image_to_text"
    WHISPER = "whisper"
    KOKORO = "kokoro"

class EngineType(str, Enum):
    """Engine used to load the model.

    Options:
    - optimum: Optimum-Intel engine
    - ovgenai: OpenVINO GenAI engine"""
    
    OV_OPTIMUM = "optimum"
    OV_GENAI = "ovgenai"
    OPENVINO = "openvino"

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


    def registered_models(self) -> dict:
        """Return only public fields as JSON-serializable dict."""
        result = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "engine": self.engine,
            "device": self.device,
            "runtime_config": self.runtime_config,
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
        # Event subscribers
        self._on_loaded: List[Callable[[ModelRecord], Awaitable[None]]] = []
        self._on_unloaded: List[Callable[[ModelRecord], Awaitable[None]]] = []

    def add_on_loaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_loaded.append(callback)

    def add_on_unloaded(self, callback: Callable[[ModelRecord], Awaitable[None]]) -> None:
        self._on_unloaded.append(callback)

    async def register_load(self, loader: ModelLoadConfig) -> str:
        # Check if model name already exists before loading
        async with self._lock:
            for existing_record in self._models.values():
                if existing_record.model_name == loader.model_name:
                    raise ValueError(f"Model name '{loader.model_name}' already registered")
        
        # Create a model record with LOADING status
        record = ModelRecord(
            model_path=loader.model_path,
            model_name=loader.model_name,
            model_type=loader.model_type,
            engine=loader.engine,
            device=loader.device,
            runtime_config=loader.runtime_config,
            status=ModelStatus.LOADING,
        )
        
        # Register the model record immediately
        async with self._lock:
            self._models[record.model_id] = record
        
        # Start background loading task
        loading_task = asyncio.create_task(self._load_task(record.model_id, loader))
        
        # Update the record with the task reference
        async with self._lock:
            if record.model_id in self._models:
                self._models[record.model_id].loading_task = loading_task
        
        return record.model_id

    async def register_unload(self, model_name: str) -> bool:
        """Unregister/unload a model by model_name. Returns True if found and unload task started."""
        async with self._lock:
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
                    record.status = ModelStatus.LOADING
                    record.loading_task = None
                else:
                    return

            # Fire loaded event callbacks outside the lock
            for cb in self._on_loaded:
                asyncio.create_task(cb(record))
                    
        except Exception as e:
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
            print(f"Error during model unload: {e}")

    async def status(self) -> dict:
        """Return registry status: total count and list of loaded models (public view)."""
        async with self._lock:
            models_public = [record.registered_models() for record in self._models.values()]
            return {
                "total_loaded_models": len(models_public),
                "models": models_public,
                "openai_model_names": [record.model_name for record in self._models.values()],
            }

async def create_model_instance(load_config: ModelLoadConfig) -> Any:
    """Factory function to create the appropriate model instance based on engine type."""
    if load_config.engine == EngineType.OV_GENAI:
        if load_config.model_type == TaskType.TEXT_TO_TEXT:
            # Import here to avoid circular imports
            from src2.engine.ov_genai.ov_genai_llm import OVGenAI_LLM
            
            model_instance = OVGenAI_LLM(load_config)
            await asyncio.to_thread(model_instance.load_model, load_config)
            return model_instance
        elif load_config.model_type == TaskType.IMAGE_TO_TEXT:
            # Import here to avoid circular imports
            from src2.engine.ov_genai.ov_genai_vlm import OVGenAI_VLM
            
            model_instance = OVGenAI_VLM(load_config)
            await asyncio.to_thread(model_instance.load_model, load_config)
            return model_instance
        elif load_config.model_type == TaskType.WHISPER:

            from src2.engine.ov_genai.whisper import OVGenAI_Whisper

            model_instance = OVGenAI_Whisper(load_config)
            await asyncio.to_thread(model_instance.load_model, load_config)
            return model_instance
            
        else:
            raise ValueError(f"Model type '{load_config.model_type}' not supported with engine '{load_config.engine}'")
    elif load_config.engine == EngineType.OPENVINO:
        if load_config.model_type == TaskType.KOKORO:
            # Import here to avoid circular imports
            from src2.engine.openvino.ov_kokoro import OV_Kokoro

            model_instance = OV_Kokoro(load_config)
            await asyncio.to_thread(model_instance.load_model, load_config)
            return model_instance
        else:
            raise ValueError(f"Model type '{load_config.model_type}' not supported with engine '{load_config.engine}'")
    elif load_config.engine == EngineType.OV_OPTIMUM:
        raise ValueError(f"Engine '{load_config.engine}' not yet implemented")

            