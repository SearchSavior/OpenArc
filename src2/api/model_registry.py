from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


async def create_model_instance(load_config: ModelLoadConfig) -> Any:
    """Factory function to create the appropriate model instance based on engine type."""
    if load_config.engine == EngineType.OV_GENAI:
        if load_config.model_type == ModelType.TEXT_TO_TEXT:
            # Import here to avoid circular imports
            from src2.engine.ov_genai.text2text import OVGenAI_Text2Text
            model_instance = OVGenAI_Text2Text(load_config)
            # Run the blocking model loading in a thread pool
            await asyncio.to_thread(model_instance.load_model, load_config)
            return model_instance
        else:
            raise ValueError(f"Model type '{load_config.model_type}' not supported with engine '{load_config.engine}'")
    elif load_config.engine == EngineType.OV_OPTIMUM:
        raise ValueError(f"Engine '{load_config.engine}' not yet implemented")
    else:
        raise ValueError(f"Unknown engine type: '{load_config.engine}'")


class ModelType(str, Enum):
    """Internal routing to the correct inference pipeline.

    Options:
    - text_to_text: Text-to-text LLM models
    - image_to_text: Image-to-text VLM models"""
    
    TEXT_TO_TEXT = "text_to_text"
    IMAGE_TO_TEXT = "image_to_text"

class EngineType(str, Enum):
    """Engine used to load the model.

    Options:
    - optimum: Optimum-Intel engine
    - ovgenai: OpenVINO GenAI engine"""
    
    OV_OPTIMUM = "optimum"
    OV_GENAI = "ovgenai"

class ModelStatus(str, Enum):
    """Status of model loading process."""
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"

class ModelLoadConfig(BaseModel):
    model_path: str = Field(
        description="""
        Top level path to directory containing OpenVINO IR converted model.
        
        OpenArc does not support runtime conversion and cannot pull from HF.
        """
    )
    model_name: str = Field(
        ...,
        description="""
        - Public facing name of the loaded model attached to a private model_id
        - Calling /v1/models will report model names from this list
        - model_name is decoupled from last segment of model_path, though in practice you should use that value.
        """
        )
    model_type: ModelType = Field(...)
    engine: EngineType = Field(...)
    device: str = Field(
        ...,
        description="""
        Device used to load the model.
        """
        )
    runtime_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional OpenVINO runtime properties."
    )

@dataclass(frozen=False, slots=True)
class ModelRecord:
    # Private fields
    model_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    time_loaded: datetime = field(default_factory=datetime.utcnow)
    model_instance: Optional[Any] = field(default=None)  # Actual loaded model instance
    status: ModelStatus = field(default=ModelStatus.LOADING)
    error_message: Optional[str] = field(default=None)

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
            "status": self.status,
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result
    

class ModelRegistry:
    """Tracks loaded models by private model_id. Async-safe."""

    def __init__(self):
        self._models: Dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()

    async def register_load(self, loader: ModelLoadConfig) -> str:
        """Register a model for loading and start the loading process in the background."""
        # Check if model name already exists
        async with self._lock:
            for existing_record in self._models.values():
                if existing_record.model_name == loader.model_name:
                    raise ValueError(f"Model name '{loader.model_name}' already registered")
        
        # Create a record in LOADING state immediately
        record = ModelRecord(
            model_path=loader.model_path,
            model_name=loader.model_name,
            model_type=loader.model_type,
            engine=loader.engine,
            device=loader.device,
            runtime_config=loader.runtime_config,
            status=ModelStatus.LOADING,
        )
        
        # Register the loading model immediately
        async with self._lock:
            self._models[record.model_id] = record
        
        # Start loading in background task
        asyncio.create_task(self._load_model_background(record.model_id, loader))
        
        return record.model_id

    async def _load_model_background(self, model_id: str, loader: ModelLoadConfig):
        """Background task to load a model and update its status."""
        try:
            # Load the model
            model_instance = await create_model_instance(loader)
            
            # Update the record with the loaded model
            async with self._lock:
                if model_id in self._models:
                    record = self._models[model_id]
                    record.model_instance = model_instance
                    record.status = ModelStatus.LOADED
                    record.error_message = None
                    
        except Exception as e:
            # Update the record with error status
            async with self._lock:
                if model_id in self._models:
                    record = self._models[model_id]
                    record.status = ModelStatus.FAILED
                    record.error_message = str(e)
                    record.model_instance = None

    async def register_unload(self, model_id: str) -> bool:
        """Unregister/unload a model by model_id. Returns True if found and removed."""
        async with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    async def get_model_instance(self, model_name: str) -> Optional[Any]:
        """Get loaded model instance by model name. Only returns if status is LOADED."""
        async with self._lock:
            for record in self._models.values():
                if record.model_name == model_name and record.status == ModelStatus.LOADED:
                    return record.model_instance
            return None

    async def get_model_record_by_name(self, model_name: str) -> Optional["ModelRecord"]:
        """Get model record by model name."""
        async with self._lock:
            for record in self._models.values():
                if record.model_name == model_name:
                    return record
            return None

    async def status(self) -> dict:
        """Return registry status: total count and list of loaded models (public view)."""
        async with self._lock:
            models_public = [record.registered_models() for record in self._models.values()]
            return {
                "total_loaded_models": len(models_public),
                "models": models_public,
            }