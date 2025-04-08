import base64
import gc
import threading
import time
import traceback
import warnings
from io import BytesIO
from typing import AsyncIterator, Optional

from optimum.intel.openvino import OVModelForVisualCausalLM
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .optimum_base_config import (
    OV_Config,
    OV_GenerationConfig,
    OV_LoadModelConfig
)

# Suppress specific deprecation warnings from optimum implementation of numpy arrays
# This block prevents clogging the API logs 
warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword")
    
class Optimum_Image2TextCore:
    """
    Loads an OpenVINO model and processor,
    Applies a chat template to conversation messages, and generates a response.
    
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
        generation_config: OV_GenerationConfig
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream generated text from an image using the provided configuration from 
        OV_GenerationConfig in completion requests.
        
        Args:
            generation_config: Configuration for text generation
                conversation: List of messages to generate text from, can include images
                max_new_tokens: Maximum number of tokens to generate
                temperature: Temperature for the model
                top_p: Top-p value for the model
                top_k: Top-k value for the model
                repetition_penalty: Repetition penalty for the model
                do_sample: Whether to sample from the model
                num_return_sequences: Number of sequences to generate
            
        Yields:
            new_text: Generated text tokens as they become available
            performance_metrics: Performance metrics for the generation
                ttft: Time to first token
                generation_time: Time taken to generate the text
                tokens_per_second: Tokens per second
                average_token_latency: Average token latency
                num_tokens_generated: Number of tokens generated
        """
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Call load_model first.")
        
        try:

            images = []
            text_conversation = []
            
            for message in generation_config.conversation:
                # Check if the message content is a list (multimodal content)
                if isinstance(message.get("content", ""), list):
                    text_parts = []
                    for content_item in message["content"]:
                        # Check if this is an image content item
                        if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                            image_url = content_item.get("image_url", {})
                            # Check if it's a base64 encoded image
                            if isinstance(image_url, dict) and image_url.get("url", "").startswith("data:image/"):
                                # Extract the base64 data
                                base64_data = image_url["url"].split(",", 1)
                                if len(base64_data) > 1:
                                    # Decode base64 to binary
                                    image_data = base64.b64decode(base64_data[1])
                                    # Convert to PIL Image
                                    image = Image.open(BytesIO(image_data))
                                    images.append(image)
                        # If it's a text content item
                        elif isinstance(content_item, dict) and content_item.get("type") == "text":
                            text_parts.append(content_item.get("text", ""))
                    
                    # Create a new message with just the text parts
                    if text_parts:
                        text_message = message.copy()
                        text_message["content"] = " ".join(text_parts)
                        text_conversation.append(text_message)
                    else:
                        # If no text parts, still include the message with empty content
                        text_message = message.copy()
                        text_message["content"] = ""
                        text_conversation.append(text_message)
                else:
                    text_conversation.append(message)
            
            text_prompt = self.processor.apply_chat_template(
                generation_config.conversation, 
                add_generation_prompt=True
            )
            
            if images:
                inputs = self.processor(
                    text=[text_prompt],
                    images=images,
                    padding=True, 
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    text=[text_prompt],
                    padding=True, 
                    return_tensors="pt"
                )
            

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
            
            first_token_received = False
            first_token_time = 0.0
            generate_start = time.perf_counter()
            thread.start()
            
            new_text = ""
            # Stream the generated text
            for new_token in streamer:
                new_text += new_token
                if not first_token_received:
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - generate_start
                    first_token_received = True
                yield new_token, {"ttft": ttft}

            thread.join()
            generate_end = time.perf_counter()

            generation_time = generate_end - generate_start
            num_tokens_generated = len(self.processor.tokenizer.encode(new_text, return_tensors="pt")[0]) - inputs.input_ids.shape[1]
            
            if generation_time > 0 and num_tokens_generated > 0:
                tokens_per_second = num_tokens_generated / generation_time
                average_token_latency = generation_time / num_tokens_generated  
                performance_metrics = {
                    "ttft": round(ttft, 2),
                    "generation_time": round(generation_time, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "average_token_latency": round(average_token_latency, 2),
                    "num_tokens_generated": num_tokens_generated,
                }
                yield new_text, performance_metrics
                
            
        except Exception as e:
            print(f"Error during vision generation: {str(e)}")
            traceback.print_exc()
            raise
        
        finally:
            if 'thread' in locals():
                thread.join()

    def util_unload_model(self):
        """Unload model and free memory"""
        del self.model
        self.model = None
        
        del self.processor
        self.processor = None
        
        gc.collect()
        print("Model unloaded and memory cleaned up")
