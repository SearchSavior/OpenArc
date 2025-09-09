#!/bin/bash

# Example curl request to load a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Impish_Nemo_12B-int4_asym-awq-ov",
    "model_name": "Impish_Nemo_12B-int4_asym-awq-ov", 
    "model_type": "text_to_text",
    "engine": "ovgenai",
    "device": "GPU.2"
  }'
