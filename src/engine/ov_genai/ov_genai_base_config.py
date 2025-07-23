
import openvino.properties.hint as ov_hints

from pydantic import BaseModel, Field
from typing import Optional



# I'm stilling working through how to build an API from this. Many other classes inherit from this 
# so pydantic models must be carefully designed to make API useful for other types of models.

class OVGenAI_Hints(BaseModel):
    execution_mode: Optional[str] = Field(
        None,
        description="Execution mode for inference. One of: 'PERFORMANCE', 'ACCURACY'."
    )
    model_distribution_policy: Optional[str] = Field(
        None,
        description="Model distribution policy. One of: 'TENSOR_PARALLEL', 'PIPELINE_PARALLEL'."
    )
    performance_mode: Optional[str] = Field(
        None,
        description="Performance mode. One of: 'LATENCY', 'THROUGHPUT', 'CUMULATIVE_THROUGHPUT'."
    )


class OVGenAI_GenerationConfig(BaseModel):
    

    do_sample: bool = Field(True, description="Whether to use sampling for generation")
    frequency_penalty: float = Field(0.0, description="Frequency penalty for token repetition")
    max_length: int = Field(..., description="Maximum length of generated tokens")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_k: int = Field(50, description="Top-k sampling parameter")
    top_p: float = Field(1.0, description="Top-p sampling parameter")