echo -e "\nSending basic chat completion request..."
IMAGE_B64=$(base64 -w 0 /home/echo/Projects/OpenArc/scripts/examples/dedication.png)

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENARC_API_KEY" \
  -d '{
    "model": "Qwen2.5-VL-3B-Instruct-int4_sym-ov",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the image in detail."},
          {"type": "image_url", "image_url": "data:image/png;base64,'"${IMAGE_B64}"'"}
        ]
      }
    ],
    "temperature": 0.8,
    "max_tokens": 256,
    "top_p": 0.9,
    "do_sample": true,
    "stream": true
  }'
