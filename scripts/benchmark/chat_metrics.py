import time

import openvino as ov
import openvino.properties.hint as ov_config
import openvino_genai

from transformers import AutoTokenizer

model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/Phi-4-mini-instruct-int4_asym-awq-se-ov"
device = "HETERO:GPU.0,GPU.2"

# Use NPU+GPU if available
pipe = openvino_genai.LLMPipeline(model_path, device=device, config={ov_config.model_distribution_policy: "PIPELINE_PARALLEL"})

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 16384

print("Chatbot ready! Type 'exit' to quit.\n")

pipe.start_chat()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    print("Bot: ", end="", flush=True)

    # Callback function for streaming
    def stream_callback(token_text: str):
        print(token_text, end="", flush=True)  # live typing
        time.sleep(0.02)  # optional delay for human-like typing effect

    # Create proper chat template using AutoTokenizer
    messages = [{"role": "user", "content": user_input}]
    prompt_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np")
    
    # Generate with streaming and capture result to access perf_metrics
    result = pipe.generate(ov.Tensor(prompt_token_ids), config, stream_callback)

    performance_report = result.perf_metrics

    performance_report = {
        'load_time (s)': round(performance_report.get_load_time() / 1000, 2), # time to load the model
        'ttft (s)': round(performance_report.get_ttft().mean / 1000, 2), # time to first token
        'tpot (ms)': round(performance_report.get_tpot().mean, 2), # time to process one token
        'throughput (tokens/s)': round(performance_report.get_throughput().mean, 2), # tokens per second (throughput)
        'generate_duration (s)': round(performance_report.get_generate_duration().mean / 1000, 2), # time to generate the response
        'input_tokens': performance_report.get_num_input_tokens(), # number of input tokens
        'new_tokens': performance_report.get_num_generated_tokens(), # number of new tokens generated
        'total_tokens': performance_report.get_num_input_tokens() + performance_report.get_num_generated_tokens(), # total number of tokens
        }

    # Formatted performance metrics
    print("\nPerformance metrics:")
    print("-" * 40)
    key_width = max(len(key) for key in performance_report.keys())
    for key, value in performance_report.items():
        print(f"  {key.ljust(key_width)} : {value}")

    print("\n")  # newline after response

pipe.finish_chat()