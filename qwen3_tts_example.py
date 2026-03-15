#!/usr/bin/env python3
"""Example script using OpenArc /v1/audio/speech for Kokoro or Qwen3 TTS.

Assumes the server is already running. Switch backends via OPENARC_TTS_BACKEND.
Uses the OpenAI Python library. Saves the generated audio to a WAV file.
"""

import os
from pathlib import Path

from openai import OpenAI

# Configuration
API_KEY = os.getenv("OPENARC_API_KEY")
BASE_URL = os.getenv("OPENARC_BASE_URL", "http://localhost:8003/v1")
# "kokoro" or "qwen3" — determines model and payload
_backend = os.getenv("OPENARC_TTS_BACKEND", "qwen3").lower()
BACKEND = _backend if _backend in ("kokoro", "qwen3") else "kokoro"

MODELS = {
    "kokoro": "kokoro",
    "qwen3": os.getenv("OPENARC_QWEN3_TTS_MODEL", "custom_voice"),
}

# Kokoro-only fields (voice=KokoroVoice, language=KokoroLanguage code)
KOKORO_CONFIG = {
    "voice": "af_sky",
    "language": "a",  # KokoroLanguage.AMERICAN_ENGLISH
    "speed": 1.0,
    "response_format": "wav",
    "character_count_chunk": 100,
}

# Qwen3 TTS only (speaker→voice, no Kokoro fields)
QWEN3_TTS_CONFIG = {
    "speaker": "uncle_fu",
    "instructions": None,
    "language": "english",
    "voice_description": None,
    "ref_audio_b64": None,
    "ref_text": None,
    "x_vector_only": False,
    "max_new_tokens": 2048,
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "non_streaming_mode": True,
    "subtalker_do_sample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
}


def generate_speech(text: str, output_path: str | Path = "speech.wav") -> Path:
    """Generate speech from text and save to WAV file.

    Uses Kokoro or Qwen3 TTS based on OPENARC_TTS_BACKEND.

    Raises:
        RuntimeError: If OPENARC_API_KEY is not set.
    """
    if not API_KEY:
        raise RuntimeError("OPENARC_API_KEY environment variable not set")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    model = MODELS[BACKEND]

    if BACKEND == "kokoro":
        cfg = KOKORO_CONFIG
        response = client.audio.speech.create(
            model=model,
            input=text,
            voice=cfg["voice"],
            extra_body={
                "language": cfg["language"],
                "speed": cfg["speed"],
                "response_format": cfg["response_format"],
                "character_count_chunk": cfg["character_count_chunk"],
            },
        )
    else:
        cfg = {k: v for k, v in QWEN3_TTS_CONFIG.items() if v is not None}
        voice = cfg.pop("speaker", "ryan")
        response = client.audio.speech.create(
            model=model,
            input=text,
            voice=voice,
            extra_body=cfg,
        )

    out = Path(output_path)
    out.write_bytes(response.content)
    return out


if __name__ == "__main__":
    text = os.getenv(
        "OPENARC_TTS_TEXT",
        "This is a test of OpenArc TTS over the API.",
    )
    out = Path(os.getenv("OPENARC_TTS_OUTPUT", "speech.wav"))

    try:
        print(f"Backend: {BACKEND}  Model: {MODELS[BACKEND]}")
        path = generate_speech(text, out)
        print(f"Saved WAV to {path}")
    except RuntimeError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"API error: {e}")
        raise SystemExit(1)
