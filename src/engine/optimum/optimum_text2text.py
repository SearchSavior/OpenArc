from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread

from typing import Any, AsyncIterator, Dict, Optional

import gc
import time
import traceback

from .optimum_base_config import (
    OV_Config, 
    OV_LoadModelConfig, 
    OV_GenerationConfig, 
    OV_PerformanceConfig
)

class Optimum_PerformanceMetrics:
    def __init__(self, performance_config: OV_PerformanceConfig):
        self.performance_config = performance_config
        self.start_time = None
        self.end_time = None
        self.eval_start_time = None
        self.eval_end_time = None
        
    def start_generation_timer(self):
        """Start the generation timer"""
        self.start_time = time.perf_counter()
        
    def stop_generation_timer(self):
        """Stop the generation timer"""
        self.end_time = time.perf_counter()
        self.performance_config.generation_time = self.end_time - self.start_time
        
    def start_eval_timer(self):
        """Start the evaluation timer"""
        self.eval_start_time = time.perf_counter()
        
    def stop_eval_timer(self):
        """Stop the evaluation timer"""
        self.eval_end_time = time.perf_counter()
        self.performance_config.eval_time = self.eval_end_time - self.eval_start_time
        
    def count_tokens(self, input_text: str, output_text: str, tokenizer: AutoTokenizer):
        """Count tokens in input and output text using the model's tokenizer"""
        try:
            # Count input tokens
            input_tokens = len(tokenizer.encode(input_text))
            self.performance_config.input_tokens = input_tokens
            
            # Count output tokens
            output_tokens = len(tokenizer.encode(output_text))
            self.performance_config.output_tokens = output_tokens
            
            # Calculate new tokens
            self.performance_config.new_tokens = output_tokens
            
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics as a dictionary"""
        return {
            "generation_time": self.performance_config.generation_time,
            "input_tokens": self.performance_config.input_tokens,
            "output_tokens": self.performance_config.output_tokens,
            "new_tokens": self.performance_config.new_tokens,
            "eval_time": self.performance_config.eval_time,
            "tokens_per_second": (self.performance_config.new_tokens / self.performance_config.generation_time) 
                if self.performance_config.generation_time and self.performance_config.new_tokens else None
        }

    def print_performance_report(self):
        """Print a formatted performance report"""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*50)
        print("INFERENCE PERFORMANCE REPORT")
        print("="*50)
        
        print(f"\nGeneration Time: {metrics['generation_time']:.3f} seconds")
        print(f"Evaluation Time: {metrics['eval_time']:.3f} seconds")
        print(f"Input Tokens: {metrics['input_tokens']}")
        print(f"Output Tokens: {metrics['output_tokens']}")
        print(f"New Tokens Generated: {metrics['new_tokens']}")
        
        if metrics['tokens_per_second']:
            print(f"Tokens/Second: {metrics['tokens_per_second']:.2f}")
            
        print("="*50)

class Optimum_Text2TextCore:
    """
    Loads an OpenVINO model and tokenizer,
    Applies a chat template to conversation messages, and generates a response.
    Exposed to the /generate endpoints.
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
        self.tokenizer = None
        self.performance_metrics = Optimum_PerformanceMetrics(OV_PerformanceConfig())
        
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

    # TODO: add performance metrics to generate_stream so we can track the same metrics in generate_text

    async def generate_stream(self, generation_config: OV_GenerationConfig) -> AsyncIterator[str]:
        """
        Asynchronously stream generated text tokens.
        
        Args:
            generation_config: Configuration for text generation containing conversation history
                             and generation parameters
        
        Yields:
            Generated text tokens as they become available
        """
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
            thread.start()

            # Stream the generated text
            for new_text in streamer:
                yield new_text

        except Exception as e:
            print(f"Error during streaming generation: {str(e)}")
            traceback.print_exc()
            raise
        
        finally:
            thread.join()

    def generate_text(self, generation_config: OV_GenerationConfig) -> tuple[str, Dict[str, Any]]:
        """
        Generate text without streaming and track performance metrics.
        
        Args:
            generation_config: Configuration for text generation containing conversation history
                             and generation parameters
        
        Returns:
            Tuple of (generated_text, performance_metrics)
        """
        try:
            # Initialize performance tracking


            # Convert conversation to input text for token counting
            input_text = "\n".join([f"{m['role']}: {m['content']}" for m in generation_config.conversation])
            
            # Start generation timer
            self.performance_metrics.start_generation_timer()

            # Generate input ids
            input_ids = self.tokenizer.apply_chat_template(
                generation_config.conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )

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
            )

            # Generate outpus from the model
            outputs = self.model.generate(**generation_kwargs)
            
            # Extract new tokens by excluding the input tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Stop generation timer
            self.performance_metrics.stop_generation_timer()
            
            # Count tokens using the model's tokenizer
            self.performance_metrics.count_tokens(
                input_text=input_text,
                output_text=generated_text,
                tokenizer=self.tokenizer
            )
            
            # Get performance metrics
            metrics = self.performance_metrics.get_performance_metrics()
            
            return generated_text, metrics

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

 
        
        
