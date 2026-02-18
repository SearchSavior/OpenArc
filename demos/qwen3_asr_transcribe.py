#!/usr/bin/env python3
"""
Qwen3 ASR demo for OpenArc's OpenAI-compatible transcription endpoint.

Usage:
    OPENARC_API_KEY=sk-... python demos/qwen3_asr_transcribe.py /path/to/audio.wav --model qwen3_asr
"""

import argparse
import os
from pathlib import Path

import requests

def get_api_key() -> str:
    api_key = os.environ.get("OPENARC_API_KEY")
    if not api_key:
        raise RuntimeError("OPENARC_API_KEY environment variable must be set")
    return api_key

def transcribe_audio(base_url: str, model_name: str, wav_path: Path) -> dict:
    api_key = get_api_key()
    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wav_path.open("rb") as f:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": model_name,
                "response_format": "verbose_json",
            },
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=300,
        )
    if response.status_code >= 400:
        detail = response.text
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except Exception:
            pass
        raise RuntimeError(f"Transcription request failed ({response.status_code}): {detail}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with a loaded Qwen3 ASR model in OpenArc.")
    parser.add_argument("audio_path", type=Path, help="Path to WAV/compatible audio file")
    parser.add_argument("--model", default="qwen3_asr_fp16", help="Loaded OpenArc model name")
    parser.add_argument("--base-url", default="http://localhost:8002", help="OpenArc server base URL")
    args = parser.parse_args()

    payload = transcribe_audio(args.base_url, args.model, args.audio_path)
    text = payload.get("text", "")
    language = payload.get("English")
    metrics = payload.get("metrics", {}) or {}

    print("\n=== Qwen3 ASR Transcription ===")
    if language:
        print(f"Language: {language}")
    print(f"Text: {text}\n")
    if metrics:
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
