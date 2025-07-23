
import openvino.properties.hint as ov_hints

from pydantic import BaseModel, Field
from typing import Optional



# I'm stilling working through how to build an API from this. Many other classes inherit from this 
# so pydantic models must be carefully designed to make API useful for other types of models.

class OV_GenAI_Hints(BaseModel):
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


class OV_GenAI_GenerationConfig(BaseModel):
    
    # adapters: Optional[AdapterConfig] = Field(None, description="Adapter configuration for LoRA")
    assistant_confidence_threshold: float = Field(..., description="Confidence threshold for assistant")
    diversity_penalty: float = Field(1.0, description="Diversity penalty for beam search")
    do_sample: bool = Field(True, description="Whether to use sampling for generation")
    echo: bool = Field(False, description="Whether to echo the prompt in the output")
    eos_token_id: int = Field(2, description="Token ID for end of sentence")
    
    frequency_penalty: float = Field(0.0, description="Frequency penalty for token repetition")
    ignore_eos: bool = Field(False, description="Whether to ignore end of sentence token")
    include_stop_str_in_output: bool = Field(False, description="Whether to include stop string in output")
    length_penalty: float = Field(1.0, description="Exponential penalty to the length for beam search")
    logprobs: int = Field(0, description="Number of top logprobs computed for each position")
    max_length: int = Field(..., description="Maximum length of generated tokens")
    max_new_tokens: int = Field(128, description="Maximum number of new tokens to generate")
    max_ngram_size: int = Field(0, description="Maximum n-gram size for no repeat n-gram")
    min_new_tokens: int = Field(0, description="Minimum number of new tokens to generate")

    no_repeat_ngram_size: int = Field(0, description="Size of n-gram to avoid repetition")
    num_assistant_tokens: int = Field(0, description="Number of assistant tokens")
    num_beam_groups: int = Field(1, description="Number of groups to divide beams into")
    num_beams: int = Field(1, description="Number of beams for beam search")
    num_return_sequences: int = Field(1, description="Number of sequences to return")
    presence_penalty: float = Field(0.0, description="Presence penalty for token repetition")
    repetition_penalty: float = Field(1.0, description="Repetition penalty for token repetition")
    rng_seed: int = Field(0, description="Random number generator seed")

  #  stop_criteria: StopCriteria = Field(..., description="Stopping criteria for beam search")
    stop_strings: set[str] = Field(set(), description="Set of strings to stop generation")
    stop_token_ids: set[int] = Field(set(), description="Set of token IDs to stop generation")

    temperature: float = Field(1.0, description="Sampling temperature")
    top_k: int = Field(50, description="Top-k sampling parameter")
    top_p: float = Field(1.0, description="Top-p sampling parameter")