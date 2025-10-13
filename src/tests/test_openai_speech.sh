#!/bin/bash


curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-82m-fp16-ov",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "af_sarah",
    "language": "a",
    "speed": 1.0,
    "response_format": "wav"
  }' \
  --output speech.wav \
  --fail