import openvino_genai as ov_genai
import time
import gc
from transformers import AutoTokenizer
from typing import List

# Define model paths
draft_model_path = r"/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/Phi-4-mini-FastDraft-120M-int8-ov"
main_model_path = r"/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/Phi-4-mini-instruct-int4_asym-awq-se-ov"

# Initialize tokenizers for accurate token counting
print("Loading tokenizers...")
try:
    # Try to load tokenizer from main model path first
    main_tokenizer = AutoTokenizer.from_pretrained(main_model_path)
    print(f"Main model tokenizer loaded: {main_tokenizer.__class__.__name__}")
except:
    # If that fails, try draft model path
    try:
        main_tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
        print(f"Using draft model tokenizer for main: {main_tokenizer.__class__.__name__}")
    except:
        raise ValueError("Could not load tokenizer from either model path")

# For consistency, we'll use the same tokenizer for both since they should be compatible
tokenizer = main_tokenizer

# Test prompts - mix of different types for comprehensive testing
test_prompts = [
    "Write a Python function to calculate fibonacci numbers:",
    "Explain the concept of machine learning in simple terms:",
    "What are the main benefits of using OpenVINO for AI inference?",
    "Create a simple REST API using Python Flask:",
    "Describe the process of photosynthesis step by step:"
]

config = ov_genai.GenerationConfig()
config.num_assistant_tokens = 5
config.max_new_tokens = 128
config.apply_chat_template = False
# config.assistant_confidence_threshold = 0.05

main_device = "GPU.0"
draft_device = "CPU"

def count_tokens(text: str) -> int:
    """Count tokens in text using the model's tokenizer"""
    return len(tokenizer.encode(text))

def warmup_pipeline(pipe: ov_genai.LLMPipeline, warmup_iterations: int = 3):
    """Warmup the pipeline to ensure stable performance measurements"""
    print(f"Running {warmup_iterations} warmup iterations...")
    warmup_prompt = "This is a warmup prompt for consistent benchmarking."
    for i in range(warmup_iterations):
        pipe.generate(warmup_prompt, config)
    print("Warmup completed.")

def benchmark_pipeline(pipe: ov_genai.LLMPipeline, prompts: List[str], name: str):
    """Benchmark pipeline with multiple prompts and return timing results"""
    print(f"\n=== {name} ===")
    
    # Warmup
    warmup_pipeline(pipe)
    
    times = []
    results = []
    token_counts = []
    
    print(f"Running {len(prompts)} test prompts...")
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Count input tokens using model tokenizer
        input_tokens = count_tokens(prompt)
        
        start_time = time.perf_counter()
        result = pipe.generate(prompt, config)
        end_time = time.perf_counter()
        
        # Count output tokens using model tokenizer
        output_tokens = count_tokens(result)
        total_tokens = input_tokens + output_tokens
        
        times.append(end_time - start_time)
        results.append(result)
        token_counts.append({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        })
        
        # Calculate model tokenizer-based throughput
        model_throughput = total_tokens / (end_time - start_time)
        
        # Show partial results
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"  Model tokenizer throughput: {model_throughput:.2f} tok/s")
        
        if hasattr(result, 'perf_metrics'):
            perf = result.perf_metrics
            print(f"  OpenVINO throughput: {perf.get_throughput().mean:.2f} tok/s")
    
    return times, results, token_counts

def calculate_stats(times: List[float]):
    """Calculate timing statistics"""
    import statistics
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }

def calculate_throughput_stats(times: List[float], token_counts: List[dict]):
    """Calculate throughput statistics using model tokenizer counts"""
    throughputs = []
    for time_val, tokens in zip(times, token_counts):
        throughput = tokens['total_tokens'] / time_val
        throughputs.append(throughput)
    
    return calculate_stats(throughputs)

def print_performance_summary(baseline_times, baseline_results, baseline_tokens, 
                            spec_times, spec_results, spec_tokens):
    """Print comprehensive performance comparison"""
    baseline_stats = calculate_stats(baseline_times)
    spec_stats = calculate_stats(spec_times)
    
    # Calculate model tokenizer-based throughput stats
    baseline_throughput_stats = calculate_throughput_stats(baseline_times, baseline_tokens)
    spec_throughput_stats = calculate_throughput_stats(spec_times, spec_tokens)
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"{'Metric':<25} {'Baseline':<15} {'Speculative':<15} {'Speedup':<10}")
    print("-" * 65)
    
    metrics = ['mean', 'median', 'min', 'max']
    for metric in metrics:
        baseline_val = baseline_stats[metric]
        spec_val = spec_stats[metric]
        speedup = baseline_val / spec_val if spec_val > 0 else 0
        
        print(f"{metric.capitalize() + ' Time (s)':<25} "
              f"{baseline_val:<15.2f} {spec_val:<15.2f} {speedup:<10.2f}x")
    
    # Model tokenizer-based throughput comparison
    print("\nModel Tokenizer-based Throughput Comparison:")
    for metric in metrics:
        baseline_val = baseline_throughput_stats[metric]
        spec_val = spec_throughput_stats[metric]
        speedup = spec_val / baseline_val if baseline_val > 0 else 0
        
        print(f"{metric.capitalize() + ' Throughput':<25} "
              f"{baseline_val:<15.2f} {spec_val:<15.2f} {speedup:<10.2f}x")
    
    # Token statistics
    print("\nToken Statistics:")
    baseline_total_tokens = sum(tc['total_tokens'] for tc in baseline_tokens)
    spec_total_tokens = sum(tc['total_tokens'] for tc in spec_tokens)
    baseline_avg_tokens = baseline_total_tokens / len(baseline_tokens)
    spec_avg_tokens = spec_total_tokens / len(spec_tokens)
    
    print(f"{'Average tokens/prompt':<25} "
          f"{baseline_avg_tokens:<15.1f} {spec_avg_tokens:<15.1f}")
    
    # Calculate average performance metrics if available
    if baseline_results and hasattr(baseline_results[0], 'perf_metrics'):
        print("\nOpenVINO Built-in Metrics:")
        baseline_throughputs = [r.perf_metrics.get_throughput().mean for r in baseline_results]
        spec_throughputs = [r.perf_metrics.get_throughput().mean for r in spec_results]
        
        avg_baseline_throughput = sum(baseline_throughputs) / len(baseline_throughputs)
        avg_spec_throughput = sum(spec_throughputs) / len(spec_throughputs)
        throughput_speedup = avg_spec_throughput / avg_baseline_throughput
        
        print(f"{'OV Avg Throughput':<25} "
              f"{avg_baseline_throughput:<15.2f} {avg_spec_throughput:<15.2f} {throughput_speedup:<10.2f}x")
        
        # TPOT comparison
        baseline_tpots = [r.perf_metrics.get_tpot().mean for r in baseline_results]
        spec_tpots = [r.perf_metrics.get_tpot().mean for r in spec_results]
        
        avg_baseline_tpot = sum(baseline_tpots) / len(baseline_tpots)
        avg_spec_tpot = sum(spec_tpots) / len(spec_tpots)
        tpot_speedup = avg_baseline_tpot / avg_spec_tpot
        
        print(f"{'OV Avg TPOT (ms/tok)':<25} "
              f"{avg_baseline_tpot:<15.2f} {avg_spec_tpot:<15.2f} {tpot_speedup:<10.2f}x")

try:
    # Benchmark WITHOUT draft model (baseline)
    print("Initializing baseline pipeline...")
    pipe_baseline = ov_genai.LLMPipeline(main_model_path, main_device)
    baseline_times, baseline_results, baseline_tokens = benchmark_pipeline(
        pipe_baseline, test_prompts, "Baseline Performance (No Draft Model)"
    )
    
    # Clean up
    del pipe_baseline
    gc.collect()
    
    # Benchmark WITH draft model (speculative decoding)
    print("\nInitializing speculative decoding pipeline...")
    draft_model = ov_genai.draft_model(draft_model_path, draft_device)
    pipe_speculative = ov_genai.LLMPipeline(
        main_model_path, 
        main_device, 
        draft_model=draft_model
    )
    
    spec_times, spec_results, spec_tokens = benchmark_pipeline(
        pipe_speculative, test_prompts, "Speculative Decoding Performance"
    )
    
    # Print comprehensive comparison
    print_performance_summary(baseline_times, baseline_results, baseline_tokens,
                            spec_times, spec_results, spec_tokens)
    
    # Clean up
    del pipe_speculative
    del draft_model
    gc.collect()
    
    print("\nBenchmarking completed successfully!")
    print(f"Tested with {len(test_prompts)} different prompts")
    print(f"Main model device: {main_device}")
    print(f"Draft model device: {draft_device}")
    print(f"Using model tokenizer: {tokenizer.__class__.__name__}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

except Exception as e:
    print(f"Error during benchmarking: {e}")
    print("Consider checking model paths and device availability")
    print("Make sure transformers is installed: pip install transformers")