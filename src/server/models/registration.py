from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from src.server.models.ov_genai import VLM_VISION_TOKENS


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


class ModelType(str, Enum):
    """
    Internal routing to the correct inference pipeline.
    
    Options:
    - llm: Text-to-text LLM models
    - vlm: Image-to-text VLM models
    - whisper: Whisper ASR models
    - kokoro: Kokoro TTS models
    - emb: Text-to-vector models    
    - rerank: Reranker models"""    
    
    LLM = "llm"
    VLM = "vlm"
    WHISPER = "whisper"
    KOKORO = "kokoro"
    EMB = "emb"
    RERANK = "rerank"


class EngineType(str, Enum):
    """Engine used to load the model.

    Options:
    - optimum: Optimum-Intel engine
    - ovgenai: OpenVINO GenAI engine"""
    
    OV_OPTIMUM = "optimum"
    OV_GENAI = "ovgenai"
    OPENVINO = "openvino"


class ModelLoadConfig(BaseModel):
    model_path: str = Field(
        description="""
        Top level path to directory containing OpenVINO IR converted model.
        
        OpenArc does not support runtime conversion and cannot pull from HF.""")
    model_name: str = Field(
        ...,
        description="""
        - Public facing name of the loaded model attached to a private model_id
        - Calling /v1/models will report loaded models by model_name.
        """
        )
    model_type: ModelType = Field(...)
    vlm_type: Optional[str] = Field(
        default=None,
        description=f"Vision token type for VLM models. Supported: {list(VLM_VISION_TOKENS.keys())}"
    )
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
    
    @field_validator('vlm_type')
    @classmethod
    def validate_vlm_type(cls, v, info):
        # Only validate if vlm_type is provided and model_type is VLM
        if v is None:
            return v
        if v not in VLM_VISION_TOKENS:
            raise ValueError(f"vlm_type must be one of {list(VLM_VISION_TOKENS.keys())}, got '{v}'")
        return v


class ModelUnloadConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to unload")

