import gc
import time
import traceback
import logging
from threading import Thread
from typing import Any, AsyncIterator, Dict, Optional

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

from .optimum_base_config import (
    OV_Config,
    OV_GenerationConfig,
    OV_LoadModelConfig,
    ModelType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Optimum_Text2TextCore:
    """
    - Initialize the Optimum_Text2TextCore class when enum ModelType (as model_type) is TEXT.
    - Loads an OpenVINO model and HuggingFace tokenizer
    - Used for text-to-text generation only
    - Any model which can be converted with the Optimum-CLI tool will work. 

    """
    def __init__(self, load_model_config: OV_LoadModelConfig, ov_config: Optional[OV_Config] = None):
        """
        Args:
            load_model_config: An instance of OV_LoadModelConfig from POST /optimum/model/load
                               
            ov_config: An instance of OV_Config from POST /optimum/model/load 
        """
        self.load_model_config = load_model_config
        self.ov_config = ov_config
        self.model = None
        self.tokenizer = None
        self.model_metadata = {
            "model_type": ModelType.TEXT,
            "id_model": load_model_config.id_model
        }

    def load_model(self):
        """Load the tokenizer and model."""
        print(f"Loading model {self.load_model_config.id_model} on device {self.load_model_config.device}...")

        # Extract its configuration as a dict 
        ov_config_dict = self.ov_config.model_dump(exclude_unset=True) if self.ov_config else {}
        
        # Load model with token IDs from config
        self.model = OVModelForCausalLM.from_pretrained(
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.load_model_config.id_model)
        print("Tokenizer loaded successfully.")

    async def generate_stream(self, generation_config: OV_GenerationConfig) -> AsyncIterator[tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """
        Asynchronously stream generated text tokens, followed by performance metrics.

        Args:
            generation_config: Configuration for text generation containing conversation history
                             and generation parameters

        Yields:
            - Tuple of (token, None) for each generated token.
            - Tuple of (None, performance_metrics) once at the end.

            performance_metrics contains
                - ttft: Time to first token
                - generation_time: Time taken to generate the text
                - tokens_per_second: Tokens per second
                - average_token_latency: Average token latency
                - num_tokens_generated: Number of tokens generated
        """

        performance_metrics = {}
        new_text = ""

        try:
            # Convert conversation to input ids using the chat template
            input_ids = self.tokenizer.apply_chat_template(
                generation_config.conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # Initialize the streamer with tokenized input
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Create generation kwargs from config
            generation_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                do_sample=generation_config.do_sample,
                repetition_penalty=generation_config.repetition_penalty,
                num_return_sequences=generation_config.num_return_sequences,
                streamer=streamer,
            )

            # Create and start the generation thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            
            first_token_received = False
            first_token_time = 0.0
            ttft = 0.0 # Initialize ttft
            generate_start = time.perf_counter()
            thread.start()

            # Stream the generated text tokens
            for new_token in streamer:
                if not first_token_received:
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - generate_start
                    first_token_received = True
                new_text += new_token
                yield new_token, None

            thread.join()
            generate_end = time.perf_counter()

            generation_time = generate_end - generate_start
            # Calculate num_tokens_generated based on the final accumulated text
            num_tokens_generated = len(self.tokenizer.encode(new_text, return_tensors="pt")[0])

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

            # Yield final metrics after streaming tokens
            yield None, performance_metrics

        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            traceback.print_exc()
            raise

    def generate_text(self, generation_config: OV_GenerationConfig) -> tuple[str, Dict[str, Any]]:
        """
        Generate text without streaming.
        (Note: This implementation currently doesn't calculate/return metrics)

        Args:
            generation_config: Configuration for text generation containing conversation history
                             and generation parameters
        Returns:
            Tuple of (generated_text, performance_metrics)
        """
        performance_metrics = {}
        try:
            input_ids = self.tokenizer.apply_chat_template(
                generation_config.conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                do_sample=generation_config.do_sample,
                repetition_penalty=generation_config.repetition_penalty,
                num_return_sequences=generation_config.num_return_sequences,
            )


            generate_start = time.perf_counter()

            outputs = self.model.generate(**generation_kwargs)

            generate_end = time.perf_counter()

            # Extract new tokens by excluding the input tokens
            new_tokens = outputs[0][input_ids.shape[1]:]

            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            generation_time = generate_end - generate_start
            num_tokens_generated = len(new_tokens)

            if generation_time > 0 and num_tokens_generated > 0:
                tokens_per_second = num_tokens_generated / generation_time
                average_token_latency = generation_time / num_tokens_generated

                performance_metrics = {
                    "generation_time": round(generation_time, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "average_token_latency": round(average_token_latency, 2),
                    "num_tokens_generated": num_tokens_generated,
                }

            return generated_text, performance_metrics

        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            traceback.print_exc()
            raise

    def util_unload_model(self):
        """Unload model and free memory"""
        del self.model
        self.model = None
        
        del self.tokenizer
        self.tokenizer = None
        
        gc.collect()
        print("Model unloaded and memory cleaned up")
