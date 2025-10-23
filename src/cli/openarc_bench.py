from transformers import AutoTokenizer
import random

def num_input_ids(model_path, num_tokens):
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

# Example usage:
#if __name__ == "__main__":
#    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Impish_Nemo_12B-int4_asym-awq-ov"
#    num_tokens = 512
#    
#    input_ids = get_input_tokens(model_path, num_tokens)
#    print(f"Generated {len(input_ids)} random tokens")
#    print(f"Sample tokens: {input_ids[:10]}")
    
