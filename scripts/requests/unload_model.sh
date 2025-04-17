#!/bin/bash

# URL of the FastAPI endpoint
API_URL="http://localhost:8000/optimum/model/unload?model_id=Qwen2.5-VL-3B-Instruct-int4_sym-ov"

# Send the DELETE request to the API
curl -X DELETE "$API_URL" -H "Authorization: Bearer $OPENARC_API_KEY"   