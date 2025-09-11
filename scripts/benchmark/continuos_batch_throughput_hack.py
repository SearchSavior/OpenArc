import time
import tiktoken

from openvino_genai import (
    GenerationConfig, 
    ContinuousBatchingPipeline, 
    SchedulerConfig,
    Tokenizer,
)

# -------------------------------------------------------------------
# Model directory
# -------------------------------------------------------------------
model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"

# -------------------------------------------------------------------
# Tokenizers
# -------------------------------------------------------------------
# tiktoken just for counting/debugging; OpenVINO uses its own Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
genai_tokenizer = Tokenizer(str(model_dir))

# -------------------------------------------------------------------
# Scheduler configuration
# -------------------------------------------------------------------
scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = 2048
scheduler_config.max_num_seqs = 12
scheduler_config.cache_size = 8
scheduler_config.dynamic_split_fuse = True
scheduler_config.enable_prefix_caching = True

# -------------------------------------------------------------------
# Continuous batching pipeline
# -------------------------------------------------------------------
pipeline = ContinuousBatchingPipeline(
    model_dir, 
    device="CPU", 
    scheduler_config=scheduler_config,
    tokenizer=genai_tokenizer,
)

# -------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant. You provide detailed, informative responses and maintain context throughout the conversation. Please give comprehensive answers that build upon previous discussion points."""

# Minimalistic filler: compresses well into tokens, predictable decoding
PADDING_PROMPT = "PAD " * 500

# -------------------------------------------------------------------
# Generation configs
# -------------------------------------------------------------------
pad_generation_config = GenerationConfig(
    max_new_tokens=512,   # keep padding completions very short
    temperature=0.1,
    top_p=0.9,
    do_sample=True
)

main_generation_config = GenerationConfig(
    max_new_tokens=512, # full-length responses for actual chat
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# -------------------------------------------------------------------
# Conversation simulation
# -------------------------------------------------------------------
conversation_turns = [
    "What are the main benefits of renewable energy?",
    "How do solar panels work exactly?",
    "What are the challenges with solar energy adoption?",
    "How does wind energy compare to solar energy?",
    "What does the future hold for renewable energy technology?"
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
    "What is the capital of Turkey?",
]

conversation_state = {
    "history": SYSTEM_PROMPT,
    "turn": 0,
    "responses": [],
    "performance_metrics": []
}

def create_batch_prompts(conversation_prompt, batch_size=20):
    """Build a batch of [main conversation + padding prompts]"""
    batch = []
    generation_configs = []
    
    # main chat
    batch.append(conversation_prompt)
    generation_configs.append(main_generation_config)
    
    # padding
    for _ in range(batch_size - 1):
        batch.append(PADDING_PROMPT)
        generation_configs.append(pad_generation_config)
    
    return batch, generation_configs

def update_conversation_state(user_input, assistant_response):
    conversation_state["history"] += f"\n\nUser: {user_input}"
    conversation_state["history"] += f"\n\nAssistant: {assistant_response}"
    conversation_state["turn"] += 1
    conversation_state["responses"].append(assistant_response)

# -------------------------------------------------------------------
# Run multi-turn simulation
# -------------------------------------------------------------------
print("Starting Multi-Turn Conversation with PAD-Based Padding")
print("=" * 70)
print(f"System Prompt: {SYSTEM_PROMPT}")
print(f"Padding Prompt: {PADDING_PROMPT[:60]}... (truncated)")
print(f"Batch Size: {scheduler_config.max_num_seqs}")
print(f"Number of Turns: {len(conversation_turns)}")
print("-" * 70)

total_start_time = time.time()

for turn_idx, user_input in enumerate(conversation_turns):
    print(f"\n🔄 TURN {turn_idx + 1}")
    print("=" * 50)
    
    # Construct conversation prompt
    current_prompt = conversation_state["history"] + f"\n\nUser: {user_input}\n\nAssistant:"
    
    # Create batch
    batch_prompts, generation_configs = create_batch_prompts(current_prompt)
    
    print(f"👤 User: {user_input}")
    
    # Count tokens
    current_tokens = len(tokenizer.encode(current_prompt))
    print(f"📝 Current conversation length: {current_tokens} tokens")
    print(f"🔢 Batch size: {len(batch_prompts)}")
    
    # Generate
    turn_start_time = time.time()
    results = pipeline.generate(batch_prompts, generation_configs)
    turn_end_time = time.time()
    
    # Extract response
    main_result = results[0]
    assistant_response = main_result.m_generation_ids[0]
    
    update_conversation_state(user_input, assistant_response)
    
    # Perf metrics
    turn_duration = turn_end_time - turn_start_time
    perf_metrics = main_result.perf_metrics
    
    ttft = perf_metrics.get_ttft().mean
    tpot = perf_metrics.get_tpot().mean
    throughput = perf_metrics.get_throughput().mean
    num_generated_tokens = perf_metrics.get_num_generated_tokens()
    
    turn_metrics = {
        "turn": turn_idx + 1,
        "turn_duration": turn_duration,
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "tokens_generated": num_generated_tokens,
        "conversation_length": current_tokens
    }
    conversation_state["performance_metrics"].append(turn_metrics)
    
    print(f"🤖 Assistant: {assistant_response[:200]}...")
    print(f"⏱️  Turn duration: {turn_duration:.2f}s")
    print(f"🚀 TTFT: {ttft/1000:.2f}s")
    print(f"📊 TPOT: {tpot:.2f}ms/token")
    print(f"🔥 Throughput: {throughput:.2f} tokens/s")
    print(f"🎯 Tokens generated: {num_generated_tokens}")
    print("-" * 50)

total_end_time = time.time()
total_duration = total_end_time - total_start_time

# -------------------------------------------------------------------
# Final conversation log
# -------------------------------------------------------------------
print("\n📚 FINAL CONVERSATION")
print("=" * 70)
print(conversation_state["history"])
print("=" * 70)

# -------------------------------------------------------------------
# Performance analysis
# -------------------------------------------------------------------
print("\n📈 PERFORMANCE ANALYSIS")
print("=" * 70)
for metrics in conversation_state["performance_metrics"]:
    print(f"Turn {metrics['turn']}: "
          f"Duration={metrics['turn_duration']:.2f}s, "
          f"TTFT={metrics['ttft']/1000:.2f}s, "
          f"Throughput={metrics['throughput']:.2f} tok/s, "
          f"ConvLen={metrics['conversation_length']} tokens")

avg_turn_duration = sum(m["turn_duration"] for m in conversation_state["performance_metrics"]) / len(conversation_state["performance_metrics"])
avg_ttft = sum(m["ttft"] for m in conversation_state["performance_metrics"]) / len(conversation_state["performance_metrics"])
avg_throughput = sum(m["throughput"] for m in conversation_state["performance_metrics"]) / len(conversation_state["performance_metrics"])
total_tokens = sum(m["tokens_generated"] for m in conversation_state["performance_metrics"])

print("\n🏆 AGGREGATE METRICS:")
print(f"Total conversation duration: {total_duration:.2f}s")
print(f"Average turn duration: {avg_turn_duration:.2f}s")
print(f"Average TTFT: {avg_ttft/1000:.2f}s")
print(f"Average throughput: {avg_throughput:.2f} tokens/s")
print(f"Total tokens generated: {total_tokens}")
print(f"Overall throughput: {total_tokens/total_duration:.2f} tokens/s")

# -------------------------------------------------------------------
# Pipeline system metrics
# -------------------------------------------------------------------
metrics = pipeline.get_metrics()
print("\n🔧 PIPELINE SYSTEM METRICS:")
print(f"Requests processed: {metrics.requests}")
print(f"Scheduled requests: {metrics.scheduled_requests}")
print(f"Cache usage: {metrics.cache_usage:.2f}%")
print(f"Max cache usage: {metrics.max_cache_usage:.2f}%")
print(f"Average cache usage: {metrics.avg_cache_usage:.2f}%")

# -------------------------------------------------------------------
# Growth analysis
# -------------------------------------------------------------------
print("\n🔍 CONVERSATION GROWTH ANALYSIS:")
first_turn_metrics = conversation_state["performance_metrics"][0]
last_turn_metrics = conversation_state["performance_metrics"][-1]

ttft_change = (last_turn_metrics["ttft"] - first_turn_metrics["ttft"]) / first_turn_metrics["ttft"] * 100
throughput_change = (last_turn_metrics["throughput"] - first_turn_metrics["throughput"]) / first_turn_metrics["throughput"] * 100
duration_change = (last_turn_metrics["turn_duration"] - first_turn_metrics["turn_duration"]) / first_turn_metrics["turn_duration"] * 100

print(f"TTFT change (Turn 1 → Turn {len(conversation_turns)}): {ttft_change:+.1f}%")
print(f"Throughput change (Turn 1 → Turn {len(conversation_turns)}): {throughput_change:+.1f}%")
print(f"Duration change (Turn 1 → Turn {len(conversation_turns)}): {duration_change:+.1f}%")
print(f"Context length growth: {first_turn_metrics['conversation_length']} → {last_turn_metrics['conversation_length']} tokens")
