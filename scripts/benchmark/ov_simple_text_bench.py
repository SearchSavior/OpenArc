from openvino_genai import LLMPipeline, GenerationConfig
#import openvino.properties.hint as ov_config
from transformers import AutoTokenizer
import openvino as ov




model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-1.7B-int8_asym-ov"

pipe = LLMPipeline(
    model_dir,       # Path to the model directory. Remember this will not pull from hub like in transformers
    device="GPU.1"
   #device="HETERO:GPU.0,GPU.2",
   #properties="PIPELINE_PARALELL"
   #device="HETERO:GPU.0,CPU",    
   #device="HETERO:GPU.1,GPU.2",
   
   #**{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}

)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

generation_config = GenerationConfig(
    max_new_tokens=32000
)

prompt = "You're the fastest Llama this side of the equator. What's your favorite food? tell me now"

messages = [{"role": "user", "content": prompt}]
# Build proper chat prompt for Qwen-style instruct models and get prompt_token_ids directly
prompt_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np")
result = pipe.generate(ov.Tensor(prompt_token_ids), generation_config=generation_config)
perf_metrics = result.perf_metrics

print(f'Load time: {perf_metrics.get_load_time() / 1000:.2f} s')
print(f'TTFT: {perf_metrics.get_ttft().mean / 1000:.2f} seconds')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
print(f'Generate duration: {perf_metrics.get_generate_duration().mean / 1000:.2f} seconds')


decoded = tokenizer.batch_decode(result.tokens, skip_special_tokens=True)
print(f"Result: {decoded[0]}")