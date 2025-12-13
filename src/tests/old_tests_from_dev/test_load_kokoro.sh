#!/bin/bash

# Example curl request to load a Kokoro TTS model via OpenArc API
curl -X POST "http://localhost:8000/openarc/load" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO",
    "model_name": "kokoro-82m-fp16-ov",
    "model_type": "kokoro",
    "engine": "openvino",
    "device": "CPU",
    "runtime_config": {}
  }'
