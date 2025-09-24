# streaming_kokoro_async.py
"""
Streaming-only Kokoro + OpenVINO implementation.
Now uses asyncio.to_thread for non-blocking streaming inference.
"""

import json
import asyncio
import gc
import re
from pathlib import Path
from typing import AsyncIterator, NamedTuple
from enum import Enum
from pydantic import BaseModel, Field

import numpy as np
import torch
import openvino as ov
import soundfile as sf

from kokoro.model import KModel
from kokoro.pipeline import KPipeline


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

class OV_KokoroLoadConfig(BaseModel):
    kokoro_path: Path = Field(..., description="Model directory containing config.json + IR")
    device: str = Field(..., description="OpenVINO device string (e.g., 'CPU', 'GPU')")

class OV_KokoroGenConfig(BaseModel):
    kokoro_message: str = Field(..., description="Text to convert to speech")
    voice: KokoroVoice = Field(..., description="Voice token from available Kokoro voices")
    lang_code: KokoroLanguage = Field(..., description="Language code for the voice")
    speed: float = Field(1.0, description="Speech speed multiplier")
    character_count_chunk: int = Field(100, description="Max characters per chunk")
    response_format: str = Field("wav", description="Output format")

class StreamChunk(NamedTuple):
    audio: torch.Tensor
    chunk_text: str
    chunk_index: int
    total_chunks: int

# =====================================================================
# Model wrapper (streaming only, async inference via to_thread)
# =====================================================================
class OV_Kokoro(KModel):
    """
    We subclass the KModel from Kokoro to use with OpenVINO inputs.
    """
    
    def __init__(self, load_config: OV_KokoroLoadConfig):
        super().__init__()
        self.model = None
        self._device = None

    def load_model(self, load_config: OV_KokoroLoadConfig):
        self.kokoro_path = load_config.kokoro_path
        self._device = load_config.device

        with (self.kokoro_path / "config.json").open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.vocab = model_config["vocab"]
        self.context_length = model_config["plbert"]["max_position_embeddings"]

        core = ov.Core()
        self.model = core.compile_model(self.kokoro_path / "openvino_model.xml", self._device)
        return self.model

    async def unload_model(self):
        self.model = None

        gc.collect()
        return True

    # -----------------------------------------------------------------
    # Text chunking
    # -----------------------------------------------------------------
    def make_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks up to `chunk_size` characters,
        preferring sentence boundaries.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Regex: split after ., !, ? followed by space
        sentences = re.split(r'(?<=[.!?]) +', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # sentence itself longer than chunk_size -> word splitting
                    words = sentence.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 > chunk_size:
                            if temp:
                                chunks.append(temp.strip())
                            temp = word
                        else:
                            temp += (" " if temp else "") + word
                    current_chunk = temp
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # -----------------------------------------------------------------
    # Streaming generator
    # -----------------------------------------------------------------
    async def chunk_forward_pass(
        self, pipeline: KPipeline, config: OV_KokoroGenConfig
    ) -> AsyncIterator[StreamChunk]:
        """
        Async generator yielding audio chunks from text.
        Uses asyncio.to_thread to offload inference calls.
        """
        text_chunks = self.make_chunks(config.kokoro_message, config.character_count_chunk)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):

            def infer_on_chunk():
                """Blocking inference run in background thread."""
                with torch.no_grad():
                    infer = pipeline(chunk_text, voice=config.voice, speed=config.speed)
                    result = next(infer) if hasattr(infer, "__iter__") else infer
                    return result

            # Run blocking inference off the main loop
            result = await asyncio.to_thread(infer_on_chunk)

            yield StreamChunk(
                audio=result.audio,
                chunk_text=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
            )


# =====================================================================
# Example usage
# =====================================================================
if __name__ == "__main__":
    load_config = OV_KokoroLoadConfig(
        kokoro_path=Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO"),
        device="CPU",
    )
    ov_model = OV_Kokoro(load_config)
    ov_model.load_model(load_config)
    pipeline = KPipeline(model=ov_model, lang_code="a")

    with open("/home/echo/Projects/OpenArc/src2/tests/test_kokoro.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        test_text = data.get("kokoro_message", "")

    gen_config = OV_KokoroGenConfig(
        kokoro_message=test_text,
        voice="af_heart",
        speed=1.0,
    )

    async def stream_example():
        print("\n--- Streaming Example ---")
        audio_chunks = []
        async for chunk in ov_model.chunk_forward_pass(pipeline, gen_config):
            print(f"[Chunk {chunk.chunk_index+1}/{chunk.total_chunks}] "
                  f"{len(chunk.chunk_text)} chars, {len(chunk.audio)} samples")
            audio_chunks.append(chunk.audio.cpu().numpy())

        if audio_chunks:
            combined = np.concatenate(audio_chunks, axis=0)
            sf.write("chunk_forward_pass_output.wav", combined, 24000)
            print("Saved chunk_forward_pass_output.wav")

    asyncio.run(stream_example())
