from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from src.server.models.optimum import PreTrainedTokenizerConfig


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: Any
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = None
    
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: Optional[bool] = None
    
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None


class OpenAIWhisperRequest(BaseModel):
    model: str
    audio_base64: Optional[str] = None  # For internal use only
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0



class OpenAISpeechRequest(BaseModel):
    """Unified request model for /v1/audio/speech; supports Kokoro and Qwen3 TTS backends."""
    # --- OpenAI standard fields ---
    model: str
    input: str
    voice: Optional[str] = None
    instructions: Optional[str] = None
    language: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    # --- Kokoro-specific ---
    character_count_chunk: Optional[int] = 100
    # --- Qwen3 TTS content ---
    voice_description: Optional[str] = None
    ref_audio_b64: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    # --- Qwen3 TTS sampling ---
    max_new_tokens: int = 2048
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    non_streaming_mode: bool = True
    subtalker_do_sample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9


# https://platform.openai.com/docs/api-reference/embeddings
class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[List[str]]]
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = "float"  # not implemented
    user: Optional[str] = None  # not implemented
    # end of openai api
    config: Optional[PreTrainedTokenizerConfig] = None


# No openai api to reference
class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    instruction: Optional[str] = None

