# URL of the FastAPI endpoint
API_URL="http://localhost:8000/optimum/generate"

# JSON payload for the generation configuration
GENERATION_CONFIG='{
    "conversation": [
        {
            "role": "system",
            "content": "You are a confused pirate muppet with a penchant for puns no one understands."
        },
        {
            "role": "user",
            "content": "Do you have the time?"
        }
    ],
    "max_new_tokens": 10000,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": true,
    "num_return_sequences": 1,
    "stream": false
}'

# Send the POST request to the API
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$GENERATION_CONFIG"
