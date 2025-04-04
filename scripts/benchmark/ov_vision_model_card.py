import time
from PIL import Image
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM


model_id = "/mnt/Ironwolf-4TB/Models/Pytorch/Qwen2.5-VL-7B-Instruct-int4_sym-ov"


ov_config = {"PERFORMANCE_HINT": "LATENCY"}

print("Loading model...")
start_load_time = time.time()
model = OVModelForVisualCausalLM.from_pretrained(model_id, export=False, device="GPU.2", ov_config=ov_config) #trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)


image_path = r"/home/echo/Projects/OpenArc/scripts/benchmark/dedication.png"
image = Image.open(image_path)
image = image.convert("RGB")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "image": image  # The image object is passed here, not just declared as a type
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

# Print number of tokens
input_token_count = len(inputs.input_ids[0])
# print(f"Input token length: {len(inputs.input_ids[0])}")

# Inference: Generation of the output with performance metrics
start_time = time.time()
output_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

num_tokens_generated = len(generated_ids[0])
load_time = time.time() - start_load_time
generation_time = time.time() - start_time
tokens_per_second = num_tokens_generated / generation_time
average_token_latency = generation_time / num_tokens_generated





print("\nPerformance Report:")
print("-"*50)
print(f"Input Tokens        : {input_token_count:>9}")
print(f"Generated Tokens    : {num_tokens_generated:>9}") # Correct variable name is num_tokens_generated
print(f"Model Load Time     : {load_time:>9.2f} sec")
print(f"Generation Time     : {generation_time:>9.2f} sec")
print(f"Throughput          : {tokens_per_second:>9.2f} t/s")
# Note: average_token_latency is average time per token, not strictly time to first token.
print(f"Avg Latency/Token   : {average_token_latency:>9.3f} sec")