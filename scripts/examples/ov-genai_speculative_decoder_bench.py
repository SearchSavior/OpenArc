import openvino_genai as ov_genai

# Define model paths
draft_model_path = r"/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"
main_model_path = r"/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/Pytorch/Llama-3.1-Nemotron-Nano-8B-v1-int4_sym-awq-se-ov"

 
prompt = "What is OpenVINO?"
 
config = ov_genai.GenerationConfig()
config.num_assistant_tokens = 28
config.max_new_tokens = 128

 
main_device = "CPU"
draft_device = "CPU"
 
draft_model = ov_genai.draft_model(draft_model_path, draft_device)
 
scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 2
 
pipe = ov_genai.LLMPipeline(
    main_model_path, 
    main_device, 
    draft_model=draft_model
    )
 
prompt = "We don't even have a chat template so strap in and let it ride!"

result = pipe.generate([prompt], generation_config=config, scheduler_config=scheduler_config)
perf_metrics = result.perf_metrics


print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'TTFT: {perf_metrics.get_ttft().mean:.2f} ms')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')

print(result)