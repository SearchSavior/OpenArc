#!/bin/bash

# API endpoint
API_URL="http://localhost:8000/optimum/model/load"

# JSON payload
JSON_PAYLOAD='{
    "load_config": {
        "id_model": "/mnt/Ironwolf-4TB/Models/OpenVINO/Rocinante-12B-v1.1-int4_sym-awq-se-ov",
        "use_cache": true,
        "device": "GPU.2",
        "export_model": false,
        "pad_token_id": null,
        "eos_token_id": null,
        "model_type": "TEXT"
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