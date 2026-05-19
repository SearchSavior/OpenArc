from pydantic import BaseModel, Field


class ContinuousBatchSchedulerConfig(BaseModel):
    max_num_batched_tokens: int = Field(default=2048, description="Maximum number of tokens to batch together")
    max_num_seqs: int = Field(default=48, description="Maximum number of sequences (batch size)")
    cache_size: int = Field(default=6, description="KV cache size in GB")
    dynamic_split_fuse: bool = Field(default=True, description="Split prompt/generate phases")
    enable_prefix_caching: bool = Field(default=True, description="Enable KV-block caching")
    use_cache_eviction: bool = Field(default=False, description="Use cache eviction")


