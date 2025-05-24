echo -e "\nSending basic chat completion request..."
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -d '{
    "model": "gemma-3-4b-it-int8_asym-ov",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Describe the image in detail."}
    ],
    "temperature": 0.8,
    "max_tokens": 256,
    "top_p": 0.9,
    "do_sample": true,
    "stream": true
}'
