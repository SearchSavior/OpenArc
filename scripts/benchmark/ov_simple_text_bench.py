import openvino_genai as ov_genai
import openvino.properties.hint as ov_config


model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"

pipe = ov_genai.LLMPipeline(
    model_dir,       # Path to the model directory
   # device="HETERO:GPU.0,GPU.1",    # Define the device to use
   # config={ov_config.model_distribution_policy: "PIPELINE_PARALLEL"}
)

generation_config = ov_genai.GenerationConfig(
    max_new_tokens=128
)

prompt = "You're the fastest Llama this side of the equator"

result = pipe.generate([prompt], generation_config=generation_config)
perf_metrics = result.perf_metrics

print(f'Load time: {perf_metrics.get_load_time() / 1000:.2f} s')
print(f'TTFT: {perf_metrics.get_ttft().mean / 1000:.2f} seconds')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
print(f'Generate duration: {perf_metrics.get_generate_duration().mean / 1000:.2f} seconds')

print(f"Result: {result}")