from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any

class ModelType(str, Enum):
    """
    Identifiers for model_type: should be extended to include other model types as OpenArc grows. 
     
     TEXT = "TEXT"
     VISION = "VISION"
    """
    TEXT = "TEXT"
    VISION = "VISION"

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
    . dynamic_shapes: whether to use dynamic shapes. Enabled by default and should not be changed expcept for special cases like NPU.

    Tokenizer specific:
    . pad_token_id: custom pad token ID
    . eos_token_id: custom end of sequence token ID
    . bos_token_id: custom beginning of sequence token ID

    Architecture specific:
    . model_type: type of model based on the architecture/task.
        - "TEXT" for text-to-text models
        - "VISION" for image-to-text models
    """
    id_model: str = Field(..., description="Model identifier or path")
    model_type: ModelType = Field(..., description="Type of model (TEXT or VISION)")
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
        conversation: Any = Field(description="A list of dicts with 'role' and 'content' keys, representing the chat history so far")
            # Any was chosen because typing is handled elsewhere and conversation dicts could contain base64 encoded images, audio files, etc.
            # Therefore a layer of pydantic is not meaninguful as we get more descriptive errors downstream.
        stream: bool = Field(False, description="Whether to stream the generated text")
        max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
        temperature: float = Field(1.0, description="Sampling temperature")
        top_k: int = Field(50, description="Top-k sampling parameter")
        top_p: float = Field(0.9, description="Top-p sampling parameter")
        repetition_penalty: float = Field(1.0, description="Repetition penalty")
        do_sample: bool = Field(True, description="Use sampling for generation")
        num_return_sequences: int = Field(1, description="Number of sequences to return")
    """
    conversation: Any = Field(description="A list of dicts with 'role' and 'content' keys, representing the chat history so far")
    stream: bool = Field(False, description="Whether to stream the generated text")
  
    # Inference parameters for generation
    max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_k: int = Field(50, description="Top-k sampling parameter")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(1.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Use sampling for generation")
    num_return_sequences: int = Field(1, description="Number of sequences to return")

def create_optimum_model(load_model_config: OV_LoadModelConfig, ov_config: Optional[OV_Config] = None):
    """
    Factory function to create the appropriate Optimum model based on configuration.
    
    Args:
        load_model_config: Configuration for loading the model
        ov_config: Optional OpenVINO configuration
        
    Returns:
        An instance of the appropriate model class (TEXT or VISION)
    
    Defines: load_model_metadata

    


    """
    # Import model classes here to avoid circular imports
    from .optimum_image2text import Optimum_Image2TextCore
    from .optimum_text2text import Optimum_Text2TextCore
    
    # Create the appropriate model instance based on configuration
    if load_model_config.model_type == ModelType.VISION:
        model_instance = Optimum_Image2TextCore(load_model_config, ov_config)
    else:
        model_instance = Optimum_Text2TextCore(load_model_config, ov_config)
    
    # Store metadata from load_model_config and ov_config in model_instance
    # This will be used for routing decisions at inference time so we can keep more than one model in memory OR on different devices.
    model_instance.model_metadata = {
        # Model configuration metadata
        "id_model": load_model_config.id_model,
        "use_cache": load_model_config.use_cache,
        "device": load_model_config.device,
        "dynamic_shapes": load_model_config.dynamic_shapes,
        "pad_token_id": load_model_config.pad_token_id,
        "eos_token_id": load_model_config.eos_token_id,
        "bos_token_id": load_model_config.bos_token_id,
        
        # Model type (now using enum)
        "model_type": load_model_config.model_type,
    }
    
    # Add OpenVINO configuration parameters if provided
    if ov_config:
        ov_config_dict = ov_config.model_dump(exclude_unset=True)
        model_instance.model_metadata.update({
            "NUM_STREAMS": ov_config_dict.get("NUM_STREAMS"),
            "PERFORMANCE_HINT": ov_config_dict.get("PERFORMANCE_HINT"),
            "PRECISION_HINT": ov_config_dict.get("PRECISION_HINT"),
            "ENABLE_HYPER_THREADING": ov_config_dict.get("ENABLE_HYPER_THREADING"),
            "INFERENCE_NUM_THREADS": ov_config_dict.get("INFERENCE_NUM_THREADS"),
            "SCHEDULING_CORE_TYPE": ov_config_dict.get("SCHEDULING_CORE_TYPE")
        })
    
    return model_instance
