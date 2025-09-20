#!/bin/bash

# Example curl request to load a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/load" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-3B-Instruct-int4_sym-ov",
    "model_name": "Qwen2.5-VL-3B-Instruct-int4_sym-ov", 
    "model_type": "image_to_text",
    "engine": "ovgenai",
    "device": "GPU.1",
    "runtime_config": {}
  }'
