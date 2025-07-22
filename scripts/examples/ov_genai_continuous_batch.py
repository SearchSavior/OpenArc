import time
from openvino_genai import (
    GenerationConfig, 
    ContinuousBatchingPipeline, 
    SchedulerConfig,
    Tokenizer,
)
# import openvino.properties.hint as ov_config


model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Llama-3.2-3B-Instruct-abliterated-OpenVINO/Llama-3.2-3B-Instruct-abliterated-int4_asym-ov"
#model_dir = "/mnt/Ironwolf-4TB/Models/Pytorch/Mistral/MS3.2-24B-Magnum-Diamond-int4_asym-ov"

genai_tokenizer = Tokenizer(str(model_dir))
# Configure scheduler with proper parameters
scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = 2048  # Maximum tokens to batch together
scheduler_config.max_num_seqs = 48  # Maximum number of sequences (batch size)
scheduler_config.cache_size = 6  # KV cache size in GB
scheduler_config.dynamic_split_fuse = True  # Split prompt/generate phases
scheduler_config.enable_prefix_caching = True  # Enable KV-block caching
#scheduler_config.use_cache_eviction = False

# Initialize the continuous batching pipeline
pipeline = ContinuousBatchingPipeline(
    model_dir, 
    device="GPU.1", 
    scheduler_config=scheduler_config,
    tokenizer=genai_tokenizer,
    #properties={}
)

# Set generation configuration
generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Example prompts for batching
prompts = [
    "You're the fastest Llama this side of the equator",
    "What is the capital of France?",
    "Explain machine learning in simple terms",
    "Write a short story about a robot",
    "Write a short story about a cat",
    "Write a short story about a dog",
    "Write a short story about a bird",
    "Write a short story about a fish",
    "Write a short story about a horse",
    "Write a short story about a cow",
    "Write a short story about a pig",
    "Write a short story about a sheep",
    "Write a short story about a goat",
    "Write a short story about a chicken",
    "Write a short story about a duck",
    "Write a short story about a goose",
    "Write a short story about a turkey",
    "Write a short story about a lion",
    "Write a short story about a tiger",
    "Write a short story about a bear",
    "Write a short story about a fox",
    "Write a short story about a wolf",
    "Write a short story about a snake",
    "Write a short story about a frog",
    "Write a short story about a turtle",
    "Write a short story about a fish",
    "Write a short story about a bird",
    "Write a short story about a dog",
    "Write a short story about a cat",
    "Write a short story about a horse",
    "Write a short story about a cow",
    "Write a short story about a pig",
    "Write a short story about a sheep",
    "Write a short story about a goat",
    "Write a short story about a chicken",
    "Write a short story about a duck",
    "Write a short story about a goose",
    "Write a short story about a turkey",
    "Write a short story about a lion",
    "Write a short story about a tiger",
    "Write a short story about a bear",
    "Write a short story about a fox",
    "Write a short story about a wolf",
    "Write a short story about a snake",
    "Write a short story about a frog",
    "Write a short story about a turtle",
    "Write a short story about a fish",
    "Write a short story about a bird",

]

# Create a list of generation configs (one for each prompt)
generation_configs = [generation_config] * len(prompts)

print("Starting continuous batching example...")
print(f"Number of prompts: {len(prompts)}")
print("-" * 50)

# Process multiple prompts in batch
start_time = time.time()

# Generate responses for all prompts
results = pipeline.generate(prompts, generation_configs)

end_time = time.time()

# Display detailed performance metrics for each result
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
    
    # Accumulate for averages
    total_tokens_generated += num_generated_tokens
    total_ttft += ttft
    total_tpot += tpot
    total_throughput += throughput
    total_generate_duration += generate_duration
    
    print(f"Prompt {i+1}: {prompts[i][:50]}...")
    print(f"  Load time: {load_time / 1000:.2f} s")
    print(f"  TTFT: {ttft / 1000:.2f} s")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Generate duration: {generate_duration / 1000:.2f} s")
    print(f"  Input tokens: {num_input_tokens}")
    print(f"  Generated tokens: {num_generated_tokens}")
    print(f"  Response: {result.m_generation_ids[0][:100]}...")
    print("-" * 80)

# Calculate and display aggregate metrics
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

# Get pipeline metrics
metrics = pipeline.get_metrics()
print("\nPipeline System Metrics:")
print("=" * 50)
print(f"Requests processed: {metrics.requests}")
print(f"Scheduled requests: {metrics.scheduled_requests}")
print(f"Cache usage: {metrics.cache_usage:.2f}%")
print(f"Max cache usage: {metrics.max_cache_usage:.2f}%")
print(f"Average cache usage: {metrics.avg_cache_usage:.2f}%")