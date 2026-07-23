from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional




class OV_Qwen3ASRGenConfig(BaseModel):
    audio_base64: str | None = Field(default=None, description="Base64 encoded audio payload (injected from file when omitted)")
    language: Optional[str] = Field(default=None, description="Optional forced language")
    max_tokens: int = Field(default=1024, description="Maximum generated tokens per chunk")
    max_chunk_sec: float = Field(default=30.0, description="Chunk size upper bound in seconds")
    search_expand_sec: float = Field(default=5.0, description="Boundary search expansion in seconds")
    min_window_ms: float = Field(default=100.0, description="Energy window in milliseconds")

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("max_chunk_sec", "search_expand_sec", "min_window_ms")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("numeric values must be positive")
        return v