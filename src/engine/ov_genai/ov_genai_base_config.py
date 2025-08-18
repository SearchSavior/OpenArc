
import openvino.properties.hint as ov_hints

from pydantic import BaseModel, Field
from typing import Optional


class OVGenAI_Hints(BaseModel):
    device: Optional[str] = Field(
        None,
        description="Device to use for inference. One of: 'GPU', 'CPU'."
    )
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
    max_new_tokens: Optional[int] = Field(None, description="Maximum length of generated tokens")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty for token repetition")
    do_sample: Optional[bool] = Field(None, description="Whether to use sampling for generation")