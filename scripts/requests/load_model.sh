#!/bin/bash

# API endpoint
API_URL="http://localhost:8000/optimum/model/load"

# JSON payload
JSON_PAYLOAD='{
    "load_config": {
        "id_model": "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama-3.1-Nemotron-Nano-8B-v1-int4_sym-awq-se-ov",
        "use_cache": true,
        "device": "GPU.1",
        "export_model": false,
        "pad_token_id": null,
        "eos_token_id": null,
        "is_vision_model": false,
        "is_text_model": true
    },
    "ov_config": {
        "NUM_STREAMS": "1",
        "PERFORMANCE_HINT": "LATENCY"
    }
}'

# Make the POST request
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENARC_API_KEY" \
    -d "$JSON_PAYLOAD"