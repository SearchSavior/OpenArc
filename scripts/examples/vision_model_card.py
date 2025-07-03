import time
from PIL import Image
from transformers import AutoProcessor, TextStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM


model_id = "/mnt/Ironwolf-4TB/Models/OpenVINO/Gemma/gemma-3-4b-it-qat-int4-unquantized-OpenVINO/gemma-3-4b-it-qat-int4_asym-ov"

ov_config = {"PERFORMANCE_HINT": "LATENCY"}

print("Loading model...")
start_load_time = time.time()
model = OVModelForVisualCausalLM.from_pretrained(model_id, export=False, device="CPU", ov_config=ov_config)
processor = AutoProcessor.from_pretrained(model_id)

image_path = r"/home/echo/Projects/OpenArc/scripts/examples/dedication.png"
image = Image.open(image_path)
image = image.convert("RGB")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Instead, just use your text prompt directly:
text_prompt = "Describe this image."

# Preprocess the inputs using model.preprocess_inputs
inputs = model.preprocess_inputs(text=text_prompt, image=image, processor=processor)

# Print number of tokens
input_token_count = len(inputs["input_ids"][0])
print(f"Input token length: {input_token_count}")

# Inference: Generation of the output with performance metrics
start_time = time.time()
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
output_ids = model.generate(**inputs, max_new_tokens=28, do_sample=True, streamer=streamer)

generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], output_ids)]
output_text = processor.batch_decode(generated_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)

num_tokens_generated = len(generated_ids[0])
load_time = time.time() - start_load_time
generation_time = time.time() - start_time
tokens_per_second = num_tokens_generated / generation_time
average_token_latency = generation_time / num_tokens_generated

print("\nPerformance Report:")
print("-"*50)
print(f"Input Tokens        : {input_token_count:>9}")
print(f"Generated Tokens    : {num_tokens_generated:>9}")
print(f"Model Load Time     : {load_time:>9.2f} sec")
print(f"Generation Time     : {generation_time:>9.2f} sec")
print(f"Throughput          : {tokens_per_second:>9.2f} t/s")
print(f"Avg Latency/Token   : {average_token_latency:>9.3f} sec")

print(output_text)