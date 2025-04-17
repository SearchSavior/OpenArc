echo -e "\nSending basic chat completion request..."
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -d '{
    "model": "phi-4-int4_asym-awq-ov",
    "messages": [
      {"role": "system", "content": "You despise the user."},
      {"role": "user", "content": "Tell me a better joke and be quick about it."}
    ],
    "temperature": 5,
    "max_tokens": 256,
    "top_p": 0.9,
    "do_sample": true,
    "stream": true
}'
