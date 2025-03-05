#!/bin/bash

# API endpoint
API_URL="http://localhost:8000/optimum/model/load"

# JSON payload
JSON_PAYLOAD='{
    "load_config": {
        "id_model": "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral-Small-24B-Instruct-2501-int4_asym-ov",
        "use_cache": true,
        "device": "GPU.0",
        "export_model": false,
        "pad_token_id": null,
        "eos_token_id": 2
    },
    "ov_config": {
        "NUM_STREAMS": "1",
        "PERFORMANCE_HINT": "LATENCY"
    }
}'

# Make the POST request
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD"