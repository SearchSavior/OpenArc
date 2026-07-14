from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional



class OV_Qwen3TTSGenConfig(BaseModel):
    """Single source of truth for all OVQwen3TTS request parameters.

    The model_type on ModelLoadConfig determines which mode the engine runs;
    supply only the fields relevant to that mode:

    - qwen3_tts_custom_voice : input, speaker, language, instruct
    - qwen3_tts_voice_design  : input, voice_description, language
    - qwen3_tts_voice_clone   : input, ref_audio_b64, ref_text, x_vector_only, language, instruct

    All modes accept the sampling fields.
    """
    # --- content ---
    input: Optional[str] = Field(default=None, description="Injected from top-level request.input by the handler; do not set here.")
    # [custom_voice]
    speaker: str | None = Field(default=None, description="[custom_voice] Predefined speaker name.")
    instruct: str | None = Field(default=None, description="[custom_voice, voice_clone] Optional style instruction.")
    # [all]
    language: str | None = Field(default=None, description="[all] Force output language. None = auto-detect.")
    # [voice_design]
    voice_description: str | None = Field(default=None, description="[voice_design] Free-form voice description.")
    # [voice_clone]
    ref_audio_b64: str | None = Field(default=None, description="[voice_clone] Base64-encoded reference WAV.")
    ref_text: str | None = Field(default=None, description="[voice_clone] Transcript of reference audio (enables ICL).")
    x_vector_only: bool = Field(default=False, description="[voice_clone] Use x-vector embedding only; skip ICL even if ref_text is set.")
    # --- sampling (all modes) ---
    max_new_tokens: int = Field(default=2048, description="Maximum codec frames to generate.")
    do_sample: bool = Field(default=True, description="Sample from logits. False = greedy.")
    top_k: int = Field(default=50, description="Top-k filter for talker logits.")
    top_p: float = Field(default=1.0, description="Nucleus filter for talker logits. 1.0 = off.")
    temperature: float = Field(default=0.9, description="Temperature scaling for talker logits.")
    repetition_penalty: float = Field(default=1.05, description="Repetition penalty on first-codebook history. 1.0 = off.")
    non_streaming_mode: bool = Field(default=True, description="True = all text tokens in prefill; False = drip-fed during decode.")
    subtalker_do_sample: bool = Field(default=True, description="Sample sub-codebook logits.")
    subtalker_top_k: int = Field(default=50, description="Top-k for code predictor.")
    subtalker_top_p: float = Field(default=1.0, description="Nucleus filter for code predictor.")
    subtalker_temperature: float = Field(default=0.9, description="Temperature for code predictor.")
    # --- streaming (HTTP: audio/L16 chunked response when stream=True) ---

    # defaults taken from https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L886
    # these apply only for the 12.5hz tokenizer model. 
    stream: bool = Field(default=True, description="Enable streaming audio output (chunked PCM).")
    stream_chunk_frames: int = Field(default=300, description="Codec frames per streaming chunk. Audio codebooks are autoregressive — each set depends on the previous — so coherent chunks require enough frames for stable prosody.")
    stream_left_context: int = Field(default=25, description="Left context frames for chunk boundary continuity (matches upstream Qwen3-TTS left_context_size=25).")
