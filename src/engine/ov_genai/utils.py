"""OV GenAI engine utilities."""
from src.server.models.ov_genai import SchedulerConfigSchema

def generate_ov_scheduler_config(scheduler_config: SchedulerConfigSchema) -> dict:
  """Generates a SchedulerConfig object from the scheduler config model.

     Note: `scheduler_config` cannot be passed to SDPA pipelines without raising
     an error. Ensure you test if the pipeline configuration is set to SDPA first
     by using methods such as `extract_scheduler_config_from_loader`.
  """
  sched_config = SchedulerConfig()
  if scheduler_config.max_num_batched_tokens:
    sched_config.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
  if scheduler_config.num_kv_blocks:
    sched_config.num_kv_blocks = scheduler_config.num_kv_blocks
  if scheduler_config.cache_size:
    sched_config.cache_size = scheduler_config.cache_size
  if scheduler_config.num_linear_attention_blocks:
    sched_config.num_linear_attention_blocks = scheduler_config.num_linear_attention_blocks
  if scheduler_config.cache_interval_multiplier:
    sched_config.cache_interval_multiplier = scheduler_config.cache_interval_multiplier
  if scheduler_config.dynamic_split_fuse:
    sched_config.dynamic_split_fuse = scheduler_config.dynamic_split_fuse
  if scheduler_config.max_num_seqs:
    sched_config.max_num_seqs = scheduler_config.max_num_seqs
  if scheduler_config.enable_prefix_caching:
    sched_config.enable_prefix_caching = scheduler_config.enable_prefix_caching
  if scheduler_config.use_cache_eviction:
    sched_config.use_cache_eviction = scheduler_config.use_cache_eviction
  if scheduler_config.use_sparse_attention:
    sched_config.use_sparse_attention = scheduler_config.use_sparse_attention
  return {"scheduler_config": sched_config}
