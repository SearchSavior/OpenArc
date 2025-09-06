from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import uuid
import asyncio

from pydantic import BaseModel, Field

from enum import Enum
from pydantic import validator

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
    
    OPTIMUM = "optimum"
    OVGNAI = "ovgenai"

class ModelLoader(BaseModel):
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

@dataclass(frozen=True, slots=True)
class ModelRecord:
    # Private fields
    model_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    time_loaded: datetime = field(default_factory=datetime.utcnow)

    # Public fields
    model_path: str = ""
    model_name: str = ""
    model_type: str = ""
    engine: str = ""
    device: str = ""
    runtime_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def internal_register(cls, create_data: ModelLoader) -> "ModelRecord":
        """Create a ModelRecord from ModelLoader data."""
        return cls(
            model_path=create_data.model_path,
            model_name=create_data.model_name,
            model_type=create_data.model_type,
            engine=create_data.engine,
            device=create_data.device,
            runtime_config=create_data.runtime_config,
        )

    def registered_models(self) -> dict:
        """Return only public fields as JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "engine": self.engine,
            "device": self.device,
            "runtime_config": self.runtime_config,
        }
    

class ModelRegistry:
    """Tracks loaded models by private model_id. Async-safe."""

    def __init__(self):
        self._models: Dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()

    async def register_load(self, loader: ModelLoader) -> str:
        record = ModelRecord.internal_register(loader)
        async with self._lock:
            self._models[record.model_id] = record
            # Ensure unique model names
            if record.model_name in self._name_to_id:
                raise ValueError(f"Model name '{record.model_name}' already registered")
            self._name_to_id[record.model_name] = record.model_id
        return record.model_id


    async def register_unload(self, model_id: str) -> bool:
        """Unregister/unload a model by model_id. Returns True if found and removed."""
        async with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    async def status(self) -> dict:
        """Return registry status: total count and list of loaded models (public view)."""
        async with self._lock:
            models_public = [record.registered_models() for record in self._models.values()]
            return {
                "total_loaded_models": len(models_public),
                "models": models_public,
            }