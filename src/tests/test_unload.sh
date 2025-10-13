#!/bin/bash

# Example curl request to unload a model via OpenArc API
curl -X POST "http://localhost:8000/openarc/unload" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Impish_Nemo_12B-int4_asym-awq-ov"
  }'
