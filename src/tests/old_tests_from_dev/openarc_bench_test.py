import random
import requests
import os

from transformers import AutoTokenizer

model_path = r"/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/dphn_Dolphin-X1-8B-int4_asym-awq-ov"
num_tokens = 512
def get_input_tokens(model_path, num_tokens):
    """
    Generate random input tokens for benchmarking.
    Follows llama.cpp approach.
    https://github.com/ggml-org/llama.cpp/blob/683fa6ba/tools/llama-bench/llama-bench.cpp#L1922
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = len(tokenizer)
    
    special_token_ids = set(tokenizer.all_special_ids)
    valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]
    
    # Generate random tokens (not repeated)
    input_ids = [random.choice(valid_token_ids) for _ in range(num_tokens)]
    
    return input_ids





response = requests.post(
    "http://localhost:8000/openarc/bench",
    headers={"Authorization": f"Bearer {os.getenv('OPENARC_API_KEY')}"},
    json={
        "model": "Dolphin-X1",
        "input_ids": get_input_tokens(model_path, num_tokens),  # Pre-encoded token IDs
        "max_tokens": 128,
        "temperature": 0.7
    }
)

metrics = response.json()
print(metrics)
# Output: {"metrics": {"input_token": ..., "new_token": ..., "ttft_ms": ..., ...}}