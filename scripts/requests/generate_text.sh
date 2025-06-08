echo -e "\nSending basic chat completion request..."
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -d '{
    "model": "Qwen3-0.6B-fp16-ov",
    "messages": [
      {"role": "system", "content": "You tell jokes about humans only another AI could understand."},
      {"role": "user", "content": "Hello, I'\''m a human. Tell me a joke."}
    ],
    "temperature": 0.8,
    "max_tokens": 256,
    "top_p": 0.9,
    "top_k": 10,
    "do_sample": true,
    "stream": true
}'
