#!/bin/bash

# URL of the FastAPI endpoint
API_URL="http://localhost:8000/optimum/model/unload?model_id=gemma-3-4b-it-int8_asym-ov"

# Send the DELETE request to the API
curl -X DELETE "$API_URL" -H "Authorization: Bearer $OPENARC_API_KEY"   


