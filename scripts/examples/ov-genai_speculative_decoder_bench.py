import openvino_genai as ov_genai
import time
import gc
from typing import List

# Define model paths
draft_model_path = r"/mnt/Ironwolf-4TB/Models/OpenVINO/Phi-4-mini-FastDraft-120M-int8-ov"
main_model_path = r"/mnt/Ironwolf-4TB/Models/OpenVINO/Phi-lthy4-OpenVINO/Phi-lthy4-int4_sym-awq-ov"

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

main_device = "GPU.0"
draft_device = "CPU"

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
    
    print(f"Running {len(prompts)} test prompts...")
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        start_time = time.perf_counter()
        result = pipe.generate(prompt, config)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        results.append(result)
        
        # Show partial results
        if hasattr(result, 'perf_metrics'):
            perf = result.perf_metrics
            print(f"  Time: {end_time - start_time:.2f}s, "
                  f"Throughput: {perf.get_throughput().mean:.2f} tok/s")
    
    return times, results

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

def print_performance_summary(baseline_times, baseline_results, spec_times, spec_results):
    """Print comprehensive performance comparison"""
    baseline_stats = calculate_stats(baseline_times)
    spec_stats = calculate_stats(spec_times)
    
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
    
    # Calculate average performance metrics if available
    if baseline_results and hasattr(baseline_results[0], 'perf_metrics'):
        print("\nThroughput Comparison:")
        baseline_throughputs = [r.perf_metrics.get_throughput().mean for r in baseline_results]
        spec_throughputs = [r.perf_metrics.get_throughput().mean for r in spec_results]
        
        avg_baseline_throughput = sum(baseline_throughputs) / len(baseline_throughputs)
        avg_spec_throughput = sum(spec_throughputs) / len(spec_throughputs)
        throughput_speedup = avg_spec_throughput / avg_baseline_throughput
        
        print(f"{'Average Throughput':<25} "
              f"{avg_baseline_throughput:<15.2f} {avg_spec_throughput:<15.2f} {throughput_speedup:<10.2f}x")
        
        # TPOT comparison
        baseline_tpots = [r.perf_metrics.get_tpot().mean for r in baseline_results]
        spec_tpots = [r.perf_metrics.get_tpot().mean for r in spec_results]
        
        avg_baseline_tpot = sum(baseline_tpots) / len(baseline_tpots)
        avg_spec_tpot = sum(spec_tpots) / len(spec_tpots)
        tpot_speedup = avg_baseline_tpot / avg_spec_tpot
        
        print(f"{'Average TPOT (ms/tok)':<25} "
              f"{avg_baseline_tpot:<15.2f} {avg_spec_tpot:<15.2f} {tpot_speedup:<10.2f}x")

try:
    # Benchmark WITHOUT draft model (baseline)
    print("Initializing baseline pipeline...")
    pipe_baseline = ov_genai.LLMPipeline(main_model_path, main_device)
    baseline_times, baseline_results = benchmark_pipeline(
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
    
    spec_times, spec_results = benchmark_pipeline(
        pipe_speculative, test_prompts, "Speculative Decoding Performance"
    )
    
    # Print comprehensive comparison
    print_performance_summary(baseline_times, baseline_results, spec_times, spec_results)
    
    # Clean up
    del pipe_speculative
    del draft_model
    gc.collect()
    
    print(f"\nBenchmarking completed successfully!")
    print(f"Tested with {len(test_prompts)} different prompts")
    print(f"Main model device: {main_device}")
    print(f"Draft model device: {draft_device}")

except Exception as e:
    print(f"Error during benchmarking: {e}")
    print("Consider checking model paths and device availability")