import openvino as ov
from openvino_genai import GenerationConfig, LLMPipeline
from transformers import AutoTokenizer

model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-REAP-25B-A3B-int4_asym-ov"

pipe = LLMPipeline(
    model_dir,       # Path to the model directory. Remember this will not pull from hub like in transformers
    device="GPU.0"

)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

generation_config = GenerationConfig(
    max_new_tokens=128
)

prompt = "You're the fastest Llama this side of the equator. What's your favorite food? try to imagine"

messages = [{"role": "user", "content": prompt}]
# Build proper chat prompt for Qwen-style instruct models and get prompt_token_ids directly
prompt_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np")

num_iterations = 3  # Number of generations to run

for i in range(num_iterations):
    print(f"\n{'='*50}")
    print(f"Generation {i+1}/{num_iterations}")
    print(f"{'='*50}")
    
    result = pipe.generate(ov.Tensor(prompt_token_ids), generation_config=generation_config)
    perf_metrics = result.perf_metrics

    print(f'Load time: {perf_metrics.get_load_time() / 1000:.2f} s')
    print(f'TTFT: {perf_metrics.get_ttft().mean / 1000:.2f} seconds')
    print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
    print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
    print(f'Generate duration: {perf_metrics.get_generate_duration().mean / 1000:.2f} seconds')

    decoded = tokenizer.batch_decode(result.tokens, skip_special_tokens=True)
    print(f"Result: {decoded[0]}")