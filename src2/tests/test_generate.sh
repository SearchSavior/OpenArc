#!/bin/bash

# Usage:
#   ./test_generate.sh [MODEL_NAME] [STREAM]
#     MODEL_NAME: name used when loading (default: Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov)
#     STREAM: true|false (default: false)

MODEL_NAME=${1:-Impish_Nemo_12B-int4_asym-awq-ov}
STREAM=${2:-true}

read -r -d '' DATA <<EOF
{
  "model_name": "${MODEL_NAME}",
  "gen_config": {
    "messages": [
      {"role": "user", "content": "Say Hello from OpenArc."}
    ],
    "max_new_tokens": 64,
    "temperature": 0.7,
    "stream": ${STREAM}
  }
}
EOF

if [ "${STREAM}" = "true" ]; then
  echo "Streaming generation (SSE) from model '${MODEL_NAME}'..."
  curl -N -sS \
    -X POST "http://localhost:8000/openarc/generate" \
    -H "Authorization: Bearer $OPENARC_API_KEY" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d "${DATA}"
else
  echo "Non-streaming generation from model '${MODEL_NAME}'..."
  curl -sS \
    -X POST "http://localhost:8000/openarc/generate" \
    -H "Authorization: Bearer $OPENARC_API_KEY" \
    -H "Content-Type: application/json" \
    -d "${DATA}"
fi


