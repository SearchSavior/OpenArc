#!/bin/bash

# API endpoint
API_URL="http://localhost:8000/optimum/model/load"

# JSON payload
JSON_PAYLOAD='{
    "load_config": {
        "id_model": "/mnt/Ironwolf-4TB/Models/OpenVINO/Phi-4-mini-instruct-int4_asym-awq-se-ov",
        "model_type": "TEXT",
        "use_cache": true,
        "device": "GPU.0",
        "dynamic_shapes": true,
        "export_model": false,
        "pad_token_id": null,
        "eos_token_id": null,
        "bos_token_id": null
    },
    "ov_config": {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT" : "null",
        "ENABLE_HYPERTHREADING" : "true",
        "INFERENCE_NUM_THREADS" : "4",
        "SCHEDULING_CORE_TYPE" : "null",
        "NUM_STREAMS" : "null"
    }
}'

# Make the POST request
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENARC_API_KEY" \
    -d "$JSON_PAYLOAD"