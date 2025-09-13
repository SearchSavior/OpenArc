#!/bin/bash

# Example curl request to test OpenAI-compatible /v1/models endpoint
curl -X GET "http://localhost:8000/v1/models" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENARC_API_KEY"