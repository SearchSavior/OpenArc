
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
