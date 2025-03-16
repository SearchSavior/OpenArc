from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any
import time




class OV_Config(BaseModel):
    """
    OpenVINO runtime optimization parameters passed as a dict in ov_config in from_pretrained.

    args:
        NUM_STREAMS: Optional[str] = Field(None, description="Number of inference streams")
        PERFORMANCE_HINT: Optional[str] = Field(None, description="LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT")
        PRECISION_HINT: Optional[str] = Field(None, description="Options: auto, fp32, fp16, int8")
        ENABLE_HYPER_THREADING: Optional[bool] = Field(None, description="Enable hyper-threading")
        INFERENCE_NUM_THREADS: Optional[int] = Field(None, description="Number of inference threads")
        SCHEDULING_CORE_TYPE: Optional[str] = Field(None, description="Options: ANY_CORE, PCORE_ONLY, ECORE_ONLY") 
    """
    NUM_STREAMS: Optional[str] = Field(None, description="Number of inference streams")
    PERFORMANCE_HINT: Optional[str] = Field(None, description="LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT")
    PRECISION_HINT: Optional[str] = Field(None, description="Options: auto, fp32, fp16, int8")
    ENABLE_HYPER_THREADING: Optional[bool] = Field(None, description="Enable hyper-threading")
    INFERENCE_NUM_THREADS: Optional[int] = Field(None, description="Number of inference threads")
    SCHEDULING_CORE_TYPE: Optional[str] = Field(None, description="Options: ANY_CORE, PCORE_ONLY, ECORE_ONLY")

class OV_LoadModelConfig(BaseModel):
    """
    Configuration for loading the model with transformers. 
    For inference:
    . id_model: model identifier or path
    . use_cache: whether to use cache for stateful models. For multi-gpu use false.
    . device: device options: CPU, GPU, AUTO
    . export_model: whether to export the model
    . dynamic_shapes: whether to use dynamic shapes.

    Tokenizer specific:
    . pad_token_id: custom pad token ID
    . eos_token_id: custom end of sequence token ID
    . bos_token_id: custom beginning of sequence token ID

    Architecture specific:
    . is_vision_model: whether the model is a vision model for image-to-text generation tasks

    """
    id_model: str = Field(..., description="Model identifier or path")
    is_vision_model: bool = Field(False, description="Whether the model is a vision model for image-to-text tasks")
    use_cache: Optional[bool] = Field(True, description="Whether to use cache for stateful models. For multi-gpu use false.")
    device: str = Field("CPU", description="Device options: CPU, GPU, AUTO")
    export_model: bool = Field(False, description="Whether to export the model")
    dynamic_shapes: Optional[bool] = Field(True, description="Whether to use dynamic shapes")
    pad_token_id: Optional[int] = Field(None, description="Custom pad token ID")
    eos_token_id: Optional[int] = Field(None, description="Custom end of sequence token ID")
    bos_token_id: Optional[int] = Field(None, description="Custom beginning of sequence token ID")
    
class OV_GenerationConfig(BaseModel):
    """
    Configuration for generation.

    args:
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]] = Field(description="A list of dicts with 'role' and 'content' keys, representing the chat history so far")
        stream: bool = Field(False, description="Whether to stream the generated text")
        max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
        temperature: float = Field(1.0, description="Sampling temperature")
        top_k: int = Field(50, description="Top-k sampling parameter")
        top_p: float = Field(0.9, description="Top-p sampling parameter")
        repetition_penalty: float = Field(1.0, description="Repetition penalty")
        do_sample: bool = Field(True, description="Use sampling for generation")
        num_return_sequences: int = Field(1, description="Number of sequences to return")
    """
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]] = Field(description="A list of dicts with 'role' and 'content' keys, representing the chat history so far")
    stream: bool = Field(False, description="Whether to stream the generated text")
  
    # Inference parameters for generation
    max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_k: int = Field(50, description="Top-k sampling parameter")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(1.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Use sampling for generation")
    num_return_sequences: int = Field(1, description="Number of sequences to return")

class OV_PerformanceConfig(BaseModel):
    '''
    Performance metrics for generation.
    
    args:
        generation_time: Optional[float] = Field(None, description="Generation time in seconds")
        input_tokens: Optional[int] = Field(None, description="Number of input tokens")
        output_tokens: Optional[int] = Field(None, description="Number of output tokens")
        new_tokens: Optional[int] = Field(None, description="Number of new tokens generated")
        eval_time: Optional[float] = Field(None, description="Evaluation time in seconds")
    '''
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens")
    new_tokens: Optional[int] = Field(None, description="Number of new tokens generated")
    eval_time: Optional[float] = Field(None, description="Evaluation time in seconds")



class Optimum_PerformanceMetrics:
    """
    Performance metrics for generation.

    args:
        performance_config: OV_PerformanceConfig = Field(description="Performance configuration")
        start_time: Optional[float] = Field(None, description="Start time")
        end_time: Optional[float] = Field(None, description="End time")
        eval_start_time: Optional[float] = Field(None, description="Evaluation start time")
        eval_end_time: Optional[float] = Field(None, description="Evaluation end time") 
    """
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

def create_optimum_model(load_model_config: OV_LoadModelConfig, ov_config: Optional[OV_Config] = None):
    """
    Factory function to create the appropriate Optimum model based on configuration.
    
    Args:
        load_model_config: Configuration for loading the model
        ov_config: Optional OpenVINO configuration
        
    Returns:
        An instance of the appropriate model class (Text2Text or Vision2Text)
    """
    if load_model_config.is_vision_model:
        # Import the vision model class only when needed
        from .optimum_image2text import Optimum_Image2TextCore
        return Optimum_Image2TextCore(load_model_config, ov_config)
    else:
        # Import here to avoid circular imports
        from .optimum_text2text import Optimum_Text2TextCore
        return Optimum_Text2TextCore(load_model_config, ov_config)



