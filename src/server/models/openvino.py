
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional



class KokoroLanguage(str, Enum):
    """Language codes for Kokoro TTS voices"""
    AMERICAN_ENGLISH = "a"
    BRITISH_ENGLISH = "b" 
    JAPANESE = "j"
    MANDARIN_CHINESE = "z"
    SPANISH = "e"
    FRENCH = "f"
    HINDI = "h"
    ITALIAN = "i"
    BRAZILIAN_PORTUGUESE = "p"

class KokoroVoice(str, Enum):
    """Available Kokoro TTS voices organized by language"""
    # American English (🇺🇸) - 11F 9M
    AF_HEART = "af_heart"
    AF_ALLOY = "af_alloy"
    AF_AOEDE = "af_aoede"
    AF_BELLA = "af_bella"
    AF_JESSICA = "af_jessica"
    AF_KORE = "af_kore"
    AF_NICOLE = "af_nicole"
    AF_NOVA = "af_nova"
    AF_RIVER = "af_river"
    AF_SARAH = "af_sarah"
    AF_SKY = "af_sky"
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"
    AM_SANTA = "am_santa"
    
    # British English (🇬🇧) - 4F 4M
    BF_ALICE = "bf_alice"
    BF_EMMA = "bf_emma"
    BF_ISABELLA = "bf_isabella"
    BF_LILY = "bf_lily"
    BM_DANIEL = "bm_daniel"
    BM_FABLE = "bm_fable"
    BM_GEORGE = "bm_george"
    BM_LEWIS = "bm_lewis"
    
    # Japanese (🇯🇵) - 4F 1M
    JF_ALPHA = "jf_alpha"
    JF_GONGITSUNE = "jf_gongitsune"
    JF_NEZUMI = "jf_nezumi"
    JF_TEBUKURO = "jf_tebukuro"
    JM_KUMO = "jm_kumo"
    
    # Mandarin Chinese (🇨🇳) - 4F 4M
    ZF_XIAOBEI = "zf_xiaobei"
    ZF_XIAONI = "zf_xiaoni"
    ZF_XIAOXIAO = "zf_xiaoxiao"
    ZF_XIAOYI = "zf_xiaoyi"
    ZM_YUNJIAN = "zm_yunjian"
    ZM_YUNXI = "zm_yunxi"
    ZM_YUNXIA = "zm_yunxia"
    ZM_YUNYANG = "zm_yunyang"
    
    # Spanish (🇪🇸) - 1F 2M
    EF_DORA = "ef_dora"
    EM_ALEX = "em_alex"
    EM_SANTA = "em_santa"
    
    # French (🇫🇷) - 1F
    FF_SIWIS = "ff_siwis"
    
    # Hindi (🇮🇳) - 2F 2M
    HF_ALPHA = "hf_alpha"
    HF_BETA = "hf_beta"
    HM_OMEGA = "hm_omega"
    HM_PSI = "hm_psi"
    
    # Italian (🇮🇹) - 1F 1M
    IF_SARA = "if_sara"
    IM_NICOLA = "im_nicola"
    
    # Brazilian Portuguese (🇧🇷) - 1F 2M
    PF_DORA = "pf_dora"
    PM_ALEX = "pm_alex"
    PM_SANTA = "pm_santa"

class OV_KokoroGenConfig(BaseModel):
    input: Optional[str] = Field(default=None, description="Injected from top-level request.input by the handler; do not set here.")
    voice: KokoroVoice = Field(KokoroVoice.AF_SARAH, description="Voice token from available Kokoro voices")
    # Optional weighted blend of voicepacks. Overrides `voice` when set.
    # Format: "af_heart,af_nicole" (equal weights) or
    #         "af_heart:0.7,af_nicole:0.3" (weights normalised by engine).
    voice_blend: Optional[str] = Field(
        default=None,
        description="Optional weighted blend of voicepacks, e.g. 'af_heart:0.7,af_nicole:0.3'. Overrides `voice`.",
    )
    lang_code: KokoroLanguage = Field(KokoroLanguage.AMERICAN_ENGLISH, description="Language code for the voice")
    speed: float = Field(1.0, description="Speech speed multiplier")
    character_count_chunk: int = Field(100, description="Max characters per chunk")
    response_format: str = Field("wav", description="Output format")

    @field_validator("voice_blend")
    @classmethod
    def _validate_voice_blend(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if not parts:
            return None
        valid = {item.value for item in KokoroVoice}
        for part in parts:
            name, _, weight = part.partition(":")
            name = name.strip()
            if name not in valid:
                raise ValueError(f"Unknown voice in blend: {name!r}")
            if weight.strip():
                try:
                    w = float(weight)
                except ValueError as exc:
                    raise ValueError(f"Invalid weight in blend for {name!r}: {weight!r}") from exc
                if w < 0:
                    raise ValueError(f"Negative weight not allowed for {name!r}: {w}")
        return v




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
