#!/usr/bin/env python3
"""
Qwen3 ASR demo for OpenArc's OpenAI-compatible transcription endpoint.

Uses the OpenAI Python library. Assumes the server is already running.

Usage:
    OPENARC_API_KEY=sk-... python demos/qwen3_asr_transcribe.py /path/to/audio.wav --model qwen3_asr
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

# Qwen3 ASR config for openarc_asr.qwen3_asr (audio_base64 injected from file)
QWEN3_ASR_CONFIG = {
    "language": None,
    "max_tokens": 1024,
    "max_chunk_sec": 30.0,
    "search_expand_sec": 5.0,
    "min_window_ms": 100.0,
}


def transcribe_audio(
    base_url: str, api_key: str, model_name: str, wav_path: Path
) -> dict:
    """Transcribe audio file using Qwen3 ASR. Returns response dict (text, metrics, etc.)."""
    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key=api_key)

    with wav_path.open("rb") as f:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            response_format="verbose_json",
            extra_body={
                "openarc_asr": json.dumps({"qwen3_asr": QWEN3_ASR_CONFIG}),
            },
        )

    return response.model_dump() if hasattr(response, "model_dump") else dict(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a loaded Qwen3 ASR model in OpenArc."
    )
    parser.add_argument("audio_path", type=Path, help="Path to WAV/compatible audio file")
    parser.add_argument(
        "--model", default="qwen3_asr", help="Loaded OpenArc model name"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8003", help="OpenArc server base URL"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENARC_API_KEY")
    if not api_key:
        raise SystemExit("OPENARC_API_KEY environment variable must be set")

    payload = transcribe_audio(args.base_url, api_key, args.model, args.audio_path)
    text = payload.get("text", "")
    language = payload.get("language")
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
