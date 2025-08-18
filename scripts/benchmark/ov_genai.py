import openvino_genai as ov_genai

# Model configuration
model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Impish_Nemo_12B-int4_asym-awq-ov"

# Initialize tokenizer and pipeline
tokenizer = ov_genai.Tokenizer(model_dir)
pipeline = ov_genai.LLMPipeline(model_dir, tokenizer, device="CPU")

# Input prompt
prompt = "Tell me a short joke about AI."

# Tokenize using OpenVINO GenAI's Tokenizer
tokenized_input = tokenizer.encode(prompt)  # Use encode() instead of direct call

# Generation parameters for beam search
generation_config = ov_genai.GenerationConfig(
    max_length=150,
    min_length=30,
    num_beams=4,
    early_stopping=True,
    top_k=100,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.0,
    num_beam_groups=2,
    diversity_penalty=0.8,
    length_penalty=1.2,
    no_repeat_ngram_size=3
)

# Generate response using tokenized input
result = pipeline.generate(tokenized_input, generation_config=generation_config)

# Decode generated token IDs to text
decoded_text = tokenizer.decode(result)

# Print generated text and performance metrics
print("Generated response:", decoded_text)
print("Generation time (s):", result)
print("Tokens generated:", result.num_tokens)
print("Tokens per second:", result.tokens_per_second)
