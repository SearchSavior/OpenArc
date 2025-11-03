#!/bin/bash

# Example curl request to load a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/load" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-0.6B-fp16-ov",
    "model_name": "Qwen3-0.6B-fp16-ov", 
    "model_type": "llm",
    "engine": "ovgenai",
    "device": "GPU.1",
    "runtime_config": {}
  }'
