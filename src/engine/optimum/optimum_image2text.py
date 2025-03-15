from transformers import AutoProcessor, TextIteratorStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM
from PIL import Image

import threading
import time
import asyncio
import traceback
import warnings


from typing import AsyncIterator, Optional

from .optimum_base_config import (
    OV_Config, 
    OV_LoadModelConfig, 
    OV_GenerationConfig, 
    OV_PerformanceConfig,
    Optimum_PerformanceMetrics
)



# Suppress specific deprecation warnings from optimum implementation of numpy arrays
# This block prevents clogging the API logs 
warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword")


    
class Optimum_Image2TextCore:
    """
    Loads an OpenVINO model and tokenizer,
    Applies a chat template to conversation messages, and generates a response.
    Exposed to the /generate endpoints.

        For OpenVINO the vision models is split into two parts:
        . language_model: The language model part of the vision model.
        . text_embeddings: The text embeddings part of the vision model.
        . vision_embeddings: The vision embeddings part of the vision model.
    """
    def __init__(self, load_model_config: OV_LoadModelConfig, ov_config: Optional[OV_Config] = None):

        """
        Args:
            load_model_config: An instance of OV_LoadModelConfig containing parameters
                               such as id_model, device, export_model, use_cache, and token IDs.
            ov_config: Optional OV_Config instance with additional model options.
        """
        self.load_model_config = load_model_config
        self.ov_config = ov_config
        self.model = None
        self.processor = None
        self.performance_metrics = Optimum_PerformanceMetrics(OV_PerformanceConfig())
        
    def load_model(self):
        """Load the tokenizer and vision model."""
        print(f"Loading model {self.load_model_config.id_model} on device {self.load_model_config.device}...")

        # Extract its configuration as a dict
        ov_config_dict = self.ov_config.model_dump(exclude_unset=True) if self.ov_config else {}

        self.model = OVModelForVisualCausalLM.from_pretrained(
            self.load_model_config.id_model,
            device=self.load_model_config.device,
            export_model=self.load_model_config.export_model,
            ov_config=ov_config_dict,
            dynamic_shapes=self.load_model_config.dynamic_shapes,
            use_cache=self.load_model_config.use_cache,
            pad_token_id=self.load_model_config.pad_token_id,
            eos_token_id=self.load_model_config.eos_token_id,
            bos_token_id=self.load_model_config.bos_token_id
        )
        print("Model loaded successfully.")

        self.processor = AutoProcessor.from_pretrained(self.load_model_config.id_model)
        print("Processor loaded successfully.")        
        
    
    async def generate_vision_stream(
        self, 
        image_path: str,
        generation_config: OV_GenerationConfig
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream generated text from an image using the provided configuration.
        
        Args:
            image_path: Path to the image file
            generation_config: Configuration for text generation
            
        Yields:
            Generated text tokens as they become available
        """
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Call load_model first.")
        
        try:
            start_time = time.time()
            
            # Load and process the image
            image = Image.open(image_path)
            
            # Add image to the conversation content
            # Find the first user message and add the image to its content
            for message in generation_config.conversation:
                if message.get("role") == "user":
                    # Check if content is a list
                    if isinstance(message.get("content"), list):
                        # Look for a place to insert the image
                        for i, content_item in enumerate(message["content"]):
                            if content_item.get("type") == "image":
                                # Replace placeholder with actual image reference
                                message["content"][i] = {"type": "image"}
                                break
                    break
            
            # Preprocess the inputs
            text_prompt = self.processor.apply_chat_template(
                generation_config.conversation, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            
            # Record input token count
            self.performance_metrics.input_tokens = len(inputs.input_ids[0])
            
            # Initialize the streamer
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Set up generation parameters
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                do_sample=generation_config.do_sample,
                num_return_sequences=generation_config.num_return_sequences,
                streamer=streamer
            )
            
            # Create and start the generation thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the generated text
            token_count = 0
            for new_text in streamer:
                token_count += 1
                yield new_text
                # Small delay to prevent blocking the event loop
                await asyncio.sleep(0.001)
            
            # Update performance metrics
            self.performance_metrics.new_tokens = token_count
            self.performance_metrics.generation_time = time.time() - start_time
            
        except Exception as e:
            print(f"Error during vision generation: {str(e)}")
            traceback.print_exc()
            raise
        
        finally:
            if 'thread' in locals():
                thread.join()

    def get_performance_metrics(self) -> OV_PerformanceConfig:
        """
        Get the performance metrics from the last generation.
        
        Returns:
            Performance metrics object
        """
        return self.performance_metrics

# model_id = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen2-VL-7B-Instruct-int4_asym-ov"
# 
# 
# ov_config = {"PERFORMANCE_HINT": "LATENCY"}
# model = OVModelForVisualCausalLM.from_pretrained(model_id, export=False, device="GPU.0", ov_config=ov_config)
# processor = AutoProcessor.from_pretrained(model_id)
# 
# 
# image_path = "/home/echo/Projects/OpenArc/scripts/examples/dedication.png"
# image = Image.open(image_path)
# 
# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]
# 
# 
# # Preprocess the inputs
# text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
# 
# inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
# 
# # Print tokenizer length
# print(f"Input token length: {len(inputs.input_ids[0])}")
# 
# # Initialize the streamer
# streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
# 
# # Set up generation parameters
# generation_kwargs = dict(
#     **inputs,
#     max_new_tokens=1024,
#     streamer=streamer
# )
# 
# # Create and start the generation thread
# thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
# thread.start()
# 
# # Stream the generated text
# print("Streaming generated text:")
# full_response = ""
# for new_text in streamer:
#     print(new_text, end="", flush=True)
#     full_response += new_text
#     time.sleep(0.01)  # Small delay to make streaming visible
# 
# print("\n\nFull response:")
# print(full_response)
# 
# # Wait for the generation to complete
# thread.join()