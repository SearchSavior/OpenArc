from openvino_genai import LLMPipeline, GenerationConfig
import openvino.properties.hint as ov_config


model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-30B-A3B-int8_asym-ov"

pipe = LLMPipeline(
    model_dir,       # Path to the model directory. Remember this will not pull from hub like in transformers
   device="HETERO:GPU.0,GPU.2",
   #properties="PIPELINE_PARALELL"
   #device="HETERO:GPU.0,CPU",    
   #device="HETERO:GPU.0,GPU.1",
   config={ov_config.model_distribution_policy: "PIPELINE_PARALLEL"}
)

generation_config = GenerationConfig(
    max_new_tokens=1024
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