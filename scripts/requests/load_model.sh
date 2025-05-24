#!/bin/bash

# API endpoint
API_URL="http://localhost:8000/optimum/model/load"

# JSON payload
JSON_PAYLOAD='{
    "load_config": {
        "id_model": "/mnt/Ironwolf-4TB/Models/OpenVINO/Phi-4-mini-instruct-int4_asym-awq-se-ov",
        "use_cache": true,
        "device": "GPU.0",
        "export_model": false,
        "pad_token_id": null,
        "eos_token_id": null,
        "model_type": "TEXT",
        "stream": false
    },
    "ov_config": {
        "PERFORMANCE_HINT": "LATENCY"
    }
}'

# Make the POST request
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENARC_API_KEY" \
    -d "$JSON_PAYLOAD"