#!/bin/bash

# Example curl request to unload a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/unload" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"
  }'
