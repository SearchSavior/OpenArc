import time
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM


model_id = "/mnt/Ironwolf-4TB/Models/Pytorch/Nous-Hermes-2-Mixtral-8x7B-DPO-int4-sym-se-ov" # Can be a local path or an HF id
ov_config = {"PERFORMANCE_HINT": "LATENCY"}

print("Loading model...")
load_time = time.perf_counter()
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=False,
    device="CPU",
    ov_config=ov_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
load_time = time.perf_counter() - load_time
print(f"Model loaded in {load_time:.3f} seconds.") 


text_prompt = "We really should join the OpenArc Discord"
conversation = [
    {
        "role": "user",
        "content": text_prompt
    }
]
text_prompt_templated = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text=text_prompt_templated, return_tensors="pt")
input_token_count = inputs['input_ids'].shape[1]

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=64)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

first_token_received = False
generate_start = 0.0
first_token = 0.0
ttft = 0.0
generated_text = ""

generate_start = time.perf_counter()
thread.start()

for new_text in streamer:
    if not first_token_received:
        first_token = time.perf_counter()
        ttft = first_token - generate_start
        first_token_received = True

    print(new_text, end='', flush=True)
    generated_text += new_text

thread.join()
generate_end = time.perf_counter()

generation_time = generate_end - generate_start

num_tokens_generated = len(tokenizer.encode(generated_text))

if generation_time > 0 and num_tokens_generated > 0:
    tokens_per_second = num_tokens_generated / generation_time
    average_token_latency = generation_time / num_tokens_generated

print("\nPerformance Report:")
print("-"*50)
print(f"Input Tokens    : {input_token_count:>9}")
print(f"Output Tokens   : {num_tokens_generated:>9}")
print("")
print(f"Load Time       : {load_time:>9.3f} sec (Model Load Time)")
print(f"TTFT            : {ttft:>9.3f} sec (Time To First Token)")
print(f"Generation Time : {generation_time:>9.3f} sec (Total Generation Time)")
print(f"Throughput      : {tokens_per_second:>9.2f} t/s (Tokens Per Second)")
print(f"Avg Latency     : {average_token_latency:>9.3f} sec (Average Token Latency)")
print("-"*50)
