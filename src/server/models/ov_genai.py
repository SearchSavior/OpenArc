from typing import Any, Dict, List

from pydantic import BaseModel, Field


class OVGenAI_GenConfig(BaseModel):
    """
    Configuration for text generation with an OpenVINO GenAI pipeline.
    Supports both text-only and multimodal (text + image) messages.
    Supports OpenAI message format including tool calls and tool responses.
    """
    messages: List[Dict[str, Any]] = Field(
        default=None,
        description="List of conversation messages. Supports OpenAI message format including user/assistant/system/tool roles, tool_calls, and tool_call_id fields."
    )
    prompt: str = Field(
        default=None,
        description="Raw text prompt (used for /v1/completions endpoint instead of messages)"
    )
    input_ids: List[int] = Field(
        default=None,
        description="Pre-encoded input token IDs (used for benchmarking to bypass tokenization)"
    )
    max_tokens: int = Field(
        default=512,
        description="""
        Maximum number of tokens to generate. OpenAI API compatible.
        OpenVINO GenAI pipeline take GenerationConfig.max_new_tokens so we have to map it to max_tokens.
        """
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature; higher values increase randomness."
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling cutoff."
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling probability cutoff."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Penalty for repeating tokens."
    )
    
    stream: bool = Field(
        default=False,
        description="Stream output in chunks of tokens."
    )
    stream_chunk_tokens: int = Field(
        default=1,
        description="Stream chunk size in tokens. Must be greater than 0. If set > 1, stream output in chunks of this many tokens using ChunkStreamer."
    )
    tools: List[Dict[str, Any]] = Field(
        default=[],
        description="List of tools/functions available to the model. Empty list by default."
    )

class OVGenAI_WhisperGenConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio")

VLM_VISION_TOKENS = {
    "internvl2": "<image>",
    "llava15": "<image>",
    "llavanext": "<image>",
    "minicpmv26": "(<image>./</image>)",
    "phi3vision": "<|image_{i}|>",
    "phi4mm": "<|image_{i}|>",
    "qwen2vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "qwen25vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "gemma3": "<start_of_image>",
}

