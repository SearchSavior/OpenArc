from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

prompt = "Alice and Bob"
checkpoint = "/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/Pytorch/Llama-3.1-Nemotron-Nano-8B-v1-int4_sym-awq-se-ov"
assistant_checkpoint = "/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

model = OVModelForCausalLM.from_pretrained(checkpoint, device="CPU", export=False)
assistant_model = OVModelForCausalLM.from_pretrained(assistant_checkpoint, device="CPU", export=False)
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)