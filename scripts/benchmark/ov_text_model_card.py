import time
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_id = "/mnt/Ironwolf-4TB/Models/OpenVINO/Ministral-8B-Instruct-2410-HF-awq-ov" # Can be a local path or an HF id

ov_config = {"PERFORMANCE_HINT": "LATENCY"}

print("Loading model...")
start_load_time = time.time()
model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=False,         # Model has already been exported to OpenVINO IR. When true exports to int8_asym
    device="GPU.0",       # In OpenVINO notation GPU.0 is first GPU device    
    ov_config=ov_config   
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

text_prompt = "We really should join the OpenArc Discord"

conversation = [
    {
        "role": "user",
        "content": text_prompt
    }
]


# Preprocess the inputs
text_prompt_templated = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


# Tokenizer converts the string into a tensor of token IDs that can be fed into the model
inputs = tokenizer(text=text_prompt_templated, return_tensors="pt")

input_token_count = inputs['input_ids'].shape[1]


print("Starting generation...")
start_time = time.time()
output_ids = model.generate(**inputs, max_new_tokens=128)
generation_time = time.time() - start_time

# Extract generated tokens by slicing the output_ids based on the input length
input_ids_tensor = inputs['input_ids']
generated_ids = output_ids[:, input_ids_tensor.shape[1]:]

# Decode the generated tokens
output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0] # Get the text from the first batch item

# Calculate performance metrics
num_tokens_generated = generated_ids.shape[1]
tokens_per_second = num_tokens_generated / generation_time
average_token_latency = generation_time / num_tokens_generated 
load_time = time.time() - start_load_time

print("\n" + "="*50)
print("Generated Text:")
print("-"*50)
print(output_text)
print("="*50)

print(""*50)
print("\nPerformance Report:")
print("-"*50)
print(f"Input Tokens        : {input_token_count:>9}")
print(f"Generated Tokens    : {num_tokens_generated:>9}")
print(""*50)
print(f"Model Load Time     : {load_time:>9.2f} sec")
print(f"Generation Time     : {generation_time:>9.2f} sec")
print(f"Throughput          : {tokens_per_second:>9.2f} t/s")
print(f"Time to first token : {average_token_latency:>9.3f} sec")


