#!/bin/bash

# Example curl request to load a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/load" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-32B-Instruct-int4_sym-awq-ov",
    "model_name": "Impish_Nemo_12B-int4_asym-awq-ov", 
    "model_type": "text_to_text",
    "engine": "ovgenai",
    "device": "HETERO:GPU.1,GPU.2",
    "runtime_config": {
      "MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"
    }
  }'
