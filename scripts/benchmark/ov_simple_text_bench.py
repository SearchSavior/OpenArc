import openvino_genai as ov_genai

model_dir = "/mnt/Ironwolf-4TB/Models/Pytorch/Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"

pipe = ov_genai.LLMPipeline(
    model_dir,  # Path to the model directory
    "CPU",    # Define the device to use
)

generation_config = ov_genai.GenerationConfig(
    max_new_tokens=128
)

prompt = "We don't even have a chat template so strap in and let it ride!"

result = pipe.generate([prompt], generation_config=generation_config)
perf_metrics = result.perf_metrics


print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'TTFT: {perf_metrics.get_ttft().mean:.2f} ms')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')

print(result)