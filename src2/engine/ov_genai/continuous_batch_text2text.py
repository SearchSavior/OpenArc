import time
from typing import List, Dict

import openvino as ov
from transformers import AutoTokenizer
from openvino_genai import (
    GenerationConfig,
    ContinuousBatchingPipeline,
    SchedulerConfig,
)

from pydantic import BaseModel, Field

class ContinuousBatchConfig(BaseModel):
    max_num_batched_tokens: int = Field(default=2048, description="Maximum number of tokens to batch together")
    max_num_seqs: int = Field(default=48, description="Maximum number of sequences (batch size)")
    cache_size: int = Field(default=6, description="KV cache size in GB")
    dynamic_split_fuse: bool = Field(default=True, description="Split prompt/generate phases")
    enable_prefix_caching: bool = Field(default=True, description="Enable KV-block caching")
    use_cache_eviction: bool = Field(default=False, description="Use cache eviction")

# -------------------------------------------------------------------
# Model directory (OpenVINO-IR export of quantized Llama 3.2 model)
# -------------------------------------------------------------------
model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-1.7B-int8_asym-ov"

# -------------------------------------------------------------------
# Initialize HuggingFace AutoTokenizer with chat template
# -------------------------------------------------------------------
encoder_tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=True,
)

# -------------------------------------------------------------------
# Helper: convert conversation messages into ov.Tensor
# -------------------------------------------------------------------
def prepare_inputs(messages: List[Dict[str, str]]) -> ov.Tensor:
    """
    Convert a chat-style messages list into an ov.Tensor
    that can be consumed by ContinuousBatchingPipeline.

    Uses HuggingFace AutoTokenizer's apply_chat_template.

    Args:
        messages (List[Dict[str,str]]): conversation turns,
            e.g. [{"role": "user", "content": "Hello"}]

    Returns:
        ov.Tensor: encoded prompt token IDs
    """
    prompt_token_ids = encoder_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,    # ensures <|assistant|> token is added
        skip_special_tokens=True,
        return_tensors="np",           # returns numpy.ndarray
    )
    return ov.Tensor(prompt_token_ids)

# -------------------------------------------------------------------
# Configure scheduler (controls batching & caching policy)
# -------------------------------------------------------------------
# Create ContinuousBatchConfig instance with default values
batch_config = ContinuousBatchConfig()

# Configure scheduler using the Pydantic model values
scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = batch_config.max_num_batched_tokens
scheduler_config.max_num_seqs = batch_config.max_num_seqs
scheduler_config.cache_size = batch_config.cache_size
scheduler_config.dynamic_split_fuse = batch_config.dynamic_split_fuse
scheduler_config.enable_prefix_caching = batch_config.enable_prefix_caching
scheduler_config.use_cache_eviction = batch_config.use_cache_eviction


# -------------------------------------------------------------------
# Continuous batching pipeline
# -------------------------------------------------------------------
pipeline = ContinuousBatchingPipeline(
    model_dir,
    device="GPU.1",
    scheduler_config=scheduler_config,
    # Note: we omit openvino_genai.Tokenizer entirely
)

# -------------------------------------------------------------------
# Generation config (applies per-prompt)
# -------------------------------------------------------------------
generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# -------------------------------------------------------------------
# Example prompts rewritten into chat-format messages
# -------------------------------------------------------------------
prompts = [
    [{"role": "user", "content": "You're the fastest Llama this side of the equator"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain machine learning in simple terms"}],
    [{"role": "user", "content": "Write a short story about a robot"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain machine learning in simple terms"}],
    [{"role": "user", "content": "Write a short story about a robot"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain machine learning in simple terms"}],
    [{"role": "user", "content": "Write a short story about a robot"}]]

# Prepare tokenized inputs as ov.Tensors
encoded_prompts = [prepare_inputs(m) for m in prompts]

# Duplicate configs for batch
generation_configs = [generation_config] * len(encoded_prompts)

print("Starting continuous batching example with AutoTokenizer...")
print(f"Number of prompts: {len(encoded_prompts)}")
print("-" * 50)

# -------------------------------------------------------------------
# Run generation
# -------------------------------------------------------------------
start_time = time.time()
results = pipeline.generate(encoded_prompts, generation_configs)
end_time = time.time()

# -------------------------------------------------------------------
# Collect metrics
# -------------------------------------------------------------------
total_tokens_generated = 0
total_ttft = 0
total_tpot = 0
total_throughput = 0
total_generate_duration = 0

print("Per-Prompt Performance Metrics:")
print("=" * 80)

for i, result in enumerate(results):
    perf_metrics = result.perf_metrics

    # Extract metrics
    load_time = perf_metrics.get_load_time()
    ttft = perf_metrics.get_ttft().mean
    tpot = perf_metrics.get_tpot().mean
    throughput = perf_metrics.get_throughput().mean
    generate_duration = perf_metrics.get_generate_duration().mean
    num_input_tokens = perf_metrics.get_num_input_tokens()
    num_generated_tokens = perf_metrics.get_num_generated_tokens()

    # Aggregate
    total_tokens_generated += num_generated_tokens
    total_ttft += ttft
    total_tpot += tpot
    total_throughput += throughput
    total_generate_duration += generate_duration

    # Note: result.m_generation_ids contains token IDs, not decoded text
    decoded_output = encoder_tokenizer.decode(
        result.m_generation_ids[0],
        skip_special_tokens=True,
    )

    print(f"Prompt {i+1}: {prompts[i][0]['content'][:50]}...")
    print(f"  Load time: {load_time / 1000:.2f} s")
    print(f"  TTFT: {ttft / 1000:.2f} s")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Generate duration: {generate_duration / 1000:.2f} s")
    print(f"  Input tokens: {num_input_tokens}")
    print(f"  Generated tokens: {num_generated_tokens}")
    print(f"  Response: {decoded_output[:200]}...")
    print("-" * 80)

# -------------------------------------------------------------------
# Aggregate metrics
# -------------------------------------------------------------------
num_prompts = len(results)
print("\nAggregate Performance Metrics:")
print("=" * 50)
print(f"Total processing time: {end_time - start_time:.2f} seconds")
print(f"Average time per prompt: {(end_time - start_time) / num_prompts:.2f} seconds")
print(f"Total tokens generated: {total_tokens_generated}")
print(f"Average TTFT: {total_ttft / num_prompts / 1000:.2f} s")
print(f"Average TPOT: {total_tpot / num_prompts:.2f} ms/token")
print(f"Average throughput: {total_throughput / num_prompts:.2f} tokens/s")
print(f"Average generate duration: {total_generate_duration / num_prompts / 1000:.2f} s")
print(f"Overall throughput: {total_tokens_generated / (end_time - start_time):.2f} tokens/s")

# -------------------------------------------------------------------
# Pipeline metrics
# -------------------------------------------------------------------
metrics = pipeline.get_metrics()
print("\nPipeline System Metrics:")
print("=" * 50)
print(f"Requests processed: {metrics.requests}")
print(f"Scheduled requests: {metrics.scheduled_requests}")
print(f"Cache usage: {metrics.cache_usage:.2f}%")
print(f"Max cache usage: {metrics.max_cache_usage:.2f}%")
print(f"Average cache usage: {metrics.avg_cache_usage:.2f}%")
