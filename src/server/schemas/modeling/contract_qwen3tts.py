from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional



class OV_Qwen3TTSGenConfig(BaseModel):
    """Base config for all OVQwen3TTS request parameters: shared sampling + streaming
    transport + the injected `input`/`language` shared by every mode.

    The loaded model's `model_type` (registration.py) determines which mode subclass the
    engine runs; supply the subclass matching that mode:

    - qwen3_tts_custom_voice (OV_Qwen3TTSCustomVoice) : input, speaker, language, instruct
    - qwen3_tts_voice_design  (OV_Qwen3TTSVoiceDesign) : input, voice_description, language
    - qwen3_tts_voice_clone   (OV_Qwen3TTSVoiceClone)  : input, ref_audio_b64, ref_text, x_vector_only, language, instruct

    All modes inherit the sampling + streaming fields below.
    """
    # --- shared content (all modes) ---
    input: Optional[str] = Field(default=None, description="Injected from top-level request.input by the handler; do not set here.")
    language: str | None = Field(default=None, description="[all] Force output language. None = auto-detect.")
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


class OV_Qwen3TTSCustomVoice(OV_Qwen3TTSGenConfig):
    """qwen3_tts_custom_voice mode: synthesize with a predefined speaker name."""
    speaker: str | None = Field(default=None, description="[custom_voice] Predefined speaker name.")
    instruct: str | None = Field(default=None, description="[custom_voice] Optional style instruction.")


class OV_Qwen3TTSVoiceDesign(OV_Qwen3TTSGenConfig):
    """qwen3_tts_voice_design mode: synthesize from a free-form voice description.

    Note: `voice_description` is fed into the engine's `instruct` slot internally; this
    config does not expose `instruct`.
    """
    voice_description: str | None = Field(default=None, description="[voice_design] Free-form voice description.")


class OV_Qwen3TTSVoiceClone(OV_Qwen3TTSGenConfig):
    """qwen3_tts_voice_clone mode: clone a reference audio's voice."""
    ref_audio_b64: str | None = Field(default=None, description="[voice_clone] Base64-encoded reference WAV.")
    ref_text: str | None = Field(default=None, description="[voice_clone] Transcript of reference audio (enables ICL).")
    x_vector_only: bool = Field(default=False, description="[voice_clone] Use x-vector embedding only; skip ICL even if ref_text is set.")
    instruct: str | None = Field(default=None, description="[voice_clone] Optional style instruction.")
