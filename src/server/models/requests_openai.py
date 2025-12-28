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



class OpenAIKokoroRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    language: Optional[str] = None
    response_format: Optional[str] = "wav"


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

