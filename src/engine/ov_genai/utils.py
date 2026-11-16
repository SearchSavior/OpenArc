"""OV GenAI engine utilities."""
import logging
from typing import Any, Literal, Mapping

from openvino_genai import GenerationConfig, SchedulerConfig, StructuredOutputConfig

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import SchedulerConfigSchema
from src.server.schemas.registration import ModelLoadConfig
from src.server.utils.structured_output import structured_output_kwargs

logger = logging.getLogger(__name__)


def apply_temperature(generation_config: GenerationConfig, temperature: float) -> None:
    """Set the sampling temperature, falling back to greedy decoding at zero.

    OpenAI treats temperature 0 as greedy, but OpenVINO GenAI rejects a
    non-positive temperature while do_sample is true, and that failure unloads
    the model.
    """
    generation_config.temperature = temperature
    if temperature <= 0:
        generation_config.do_sample = False


def build_structured_output_config(
    response_format: Mapping[str, Any] | None,
) -> StructuredOutputConfig | None:
    """Build the OpenVINO GenAI structured-output config for a request, if any.

    Returns ``None`` when the request asks for no constraint, so callers leave
    the generation config's own default untouched.
    """
    kwargs = structured_output_kwargs(response_format)
    return StructuredOutputConfig(**kwargs) if kwargs else None


def prefill_metrics(perf_metrics) -> tuple[float, int, int]:
    """Return (prefill tokens/s, newly processed tokens, prefix-cache hit tokens).

    OpenVINO GenAI reports the prompt length as tokenized (N) and TTFT as the
    time to produce the first token, which only covers the tokens the pipeline
    actually recomputed. When KV is reused from an earlier request, N - K tokens
    were copied rather than computed, so dividing N by TTFT inflates the rate by
    the cache-hit ratio. Subtracting the hit count keeps the numerator and the
    denominator describing the same work.

    The hit accessor is feature-detected because we track nightly wheels: a
    build without it leaves the hit count at 0, so callers must surface it in
    the reported metrics to tell a cold run from an unreported one.
    """
    hit_fn = getattr(perf_metrics, "get_num_prefix_cache_hit_tokens", None)
    hit = hit_fn() if hit_fn else 0
    total = perf_metrics.get_num_input_tokens()
    new_tokens = total - hit
    ttft_s = perf_metrics.get_ttft().mean / 1000
    return (round(new_tokens / ttft_s, 2) if ttft_s > 0 else 0.0), new_tokens, hit


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

def extract_scheduler_config_from_loader(loader: ModelLoadConfig) -> dict[Literal["scheduler_config"], SchedulerConfig]:
  """Extract the scheduler configuration from the loader config and return as a dict to be piped to the pipeline.

     If pipeline is SDPA, returns an empty dictionary and raises an error to the user.
     Otherwise, returns a dictonary with the SchedulerConfig object
  """
  pipeline_kwargs = loader.runtime_config or {}
  sched_config = loader.scheduler_config or SchedulerConfigSchema()
  sched_config_dict = sched_config.model_dump(exclude_unset=True)
  if pipeline_kwargs.get("ATTENTION_BACKEND") == "SDPA" and sched_config_dict:
    logger.error("Cannot set scheduler_config for model: scheduler config is unsupported for SDPA backends")
    return {}
  if not sched_config_dict:
    return {}
  return generate_ov_scheduler_config(sched_config)
