#!/bin/bash

# Example curl request to check OpenArc API status
curl -X GET "http://localhost:8000/openarc/status" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -H "Content-Type: application/json" \
