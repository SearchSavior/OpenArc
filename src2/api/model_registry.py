from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import asyncio
from pydantic import BaseModel, Field

# Only ONE Pydantic model for input validation
class ModelRecordCreate(BaseModel):
    model_path: str = Field(..., 
        description="""
        Path to the model directory (top-level).
        """)
    model_name: str = Field(..., 
        description="""
        Public facing name of the loaded model attached to a private _id_model.

        Calling /v1/models will report model names from this list.

        model_name is seperate from last segment of model_path, though in practice you should use that value
        
        """)
    model_type: str = Field (
        options=["text_to_text", "image_to_text"],
        description="""
        Internal identifier for routing to the correct inference pipeline.

        args:
        - text_to_text: Text-to-text LLM models
        - image_to_text: Image-to-text VLM models
        """)
    engine: str = Field(
        options=["optimum", "ovgenai"],
        description="""
        Engine used to load the model.

        args:
        - optimum: Optimum-Intel engine
        - ovgenai: OpenVINO GenAI engine
        """)
    device: str = Field(..., 
        description="""
        Device used to load the model.

        """)
        
    properties: Dict[str, Any] = Field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ModelRecord:
    # Private fields
    _model_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _time_loaded: datetime = field(default_factory=datetime.utcnow)
    
    # Public fields
    model_path: str = ""
    model_name: str = ""
    model_type: str = ""
    engine: str = ""
    device: str = ""
    properties: Dict[str, Any] = field()

    @classmethod
    def internal_register(cls, create_data: ModelRecordCreate) -> "ModelRecord":
        return cls(**create_data.dict())

    def to_public_json(self) -> dict:
        """Return only public fields as JSON-serializable dict"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "engine": self.engine,
            "device": self.device,
            "properties": self.properties,
        }