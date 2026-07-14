from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class OVGenAI_WhisperGenConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio")
