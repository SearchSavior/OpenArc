"""OV GenAI engine utilities."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import openvino_genai
from openvino_genai import GenerationConfig, SchedulerConfig

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import (
    OVGenAI_GenConfig,
    SchedulerConfigSchema,
)
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)


def format_perf_metrics(gen_config: OVGenAI_GenConfig, perf_metrics) -> Dict[str, Any]:
  """Collect and format performance metrics into a dictionary.

     Shared by the LLM, VLM, and continuous-batching engines.

     Args:
         gen_config: OVGenAI_GenConfig
         perf_metrics: PerfMetrics

     Returns:
         metrics: Dict[str, Any]
  """
  # Compute prefill throughput = input tokens / ttft (in seconds)
  # Inspired by section 2.2 (https://arxiv.org/pdf/2404.14294v3)
  ttft_seconds = perf_metrics.get_ttft().mean / 1000
  input_tokens = perf_metrics.get_num_input_tokens()
  prefill_throughput = round(input_tokens / ttft_seconds, 2) if ttft_seconds > 0 else 0

  metrics: Dict[str, Any] = {
    'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
    'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
    'tpot (ms)': round(perf_metrics.get_tpot().mean, 5),
    'prefill_throughput (tokens/s)': prefill_throughput,
    'decode_throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 5),
    'decode_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 5),
    'input_token': input_tokens,
    'new_token': perf_metrics.get_num_generated_tokens(),
    'total_token': input_tokens + perf_metrics.get_num_generated_tokens(),
    'stream': gen_config.stream,
  }
  # Include streaming-specific fields
  if gen_config.stream and hasattr(gen_config, "stream_chunk_tokens"):
    metrics['stream_chunk_tokens'] = gen_config.stream_chunk_tokens

  return metrics


def load_draft_model(loader: ModelLoadConfig) -> Tuple[Any, bool, Optional[int], Optional[float]]:
  """Load the speculative-decoding draft model configured on the loader.

     Returns (draft_model, draft_model_loaded, num_assistant_tokens,
     assistant_confidence_threshold). draft_model is None when no draft model
     is configured or when loading fails (a warning is logged and generation
     continues without speculation). Exactly one of num_assistant_tokens /
     assistant_confidence_threshold is set (XOR requirement).

     The draft model is cached alongside the main model: OpenVINO keys cache
     blobs by model content, so sharing one CACHE_DIR is safe.
  """
  if not loader.draft_model_path:
    return None, False, None, None

  draft_model = None
  draft_model_loaded = False
  num_assistant_tokens = None
  assistant_confidence_threshold = None
  try:
    draft_model_properties = {}
    if loader.cache_dir:
      draft_model_properties['CACHE_DIR'] = loader.cache_dir
    draft_model = openvino_genai.draft_model(
      loader.draft_model_path,
      loader.draft_device,
      **draft_model_properties
    )
    logger.info(f"Loaded draft model from {loader.draft_model_path} on {loader.draft_device}")
    draft_model_loaded = True

    # Ensure we always have exactly one parameter set (XOR requirement)
    if loader.num_assistant_tokens is not None:
      num_assistant_tokens = loader.num_assistant_tokens
    elif loader.assistant_confidence_threshold is not None:
      assistant_confidence_threshold = loader.assistant_confidence_threshold
    else:
      default_tokens = int(os.getenv('OPENARC_DEFAULT_NUM_ASSISTANT_TOKENS', '4'))
      num_assistant_tokens = default_tokens
      logger.info(f"Using default num_assistant_tokens={default_tokens} for speculative decoding")
  except Exception as e:
    logger.warning(f"Failed to load draft model: {e}, continuing without speculative decoding")
    draft_model = None
    draft_model_loaded = False
    num_assistant_tokens = None
    assistant_confidence_threshold = None

  return draft_model, draft_model_loaded, num_assistant_tokens, assistant_confidence_threshold


def apply_temperature(generation_config: GenerationConfig, temperature: float) -> None:
    """Set the sampling temperature, falling back to greedy decoding at zero.

    OpenAI treats temperature 0 as greedy, but OpenVINO GenAI rejects a
    non-positive temperature while do_sample is true, and that failure unloads
    the model.
    """
    generation_config.temperature = temperature
    if temperature <= 0:
        generation_config.do_sample = False


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
  if pipeline_kwargs.get("ATTENTION_BACKEND") == "SDPA":
    if sched_config_dict:
      logger.error("Cannot set scheduler_config for model: scheduler config is unsupported for SDPA backends")
    return {}
  # Arc-tuned default for the PA backend: prefix caching keeps the KV blocks
  # of earlier prompts around, which pays off on every client that resends
  # long histories (chat UIs, agents). Author enable_prefix_caching: false in
  # the model's scheduler_config to opt out.
  if "enable_prefix_caching" not in sched_config_dict:
    sched_config.enable_prefix_caching = True
  # Prefill chunk tuning, swept on an Arc Pro B70 (8K prompt, Qwen3.8-27B
  # int4, cold prefill): max_num_batched_tokens 256 -> 8192 with
  # dynamic_split_fuse raised prefill from 1464 to 1968 tok/s (TTFT -26%).
  # dynamic_split_fuse must stay on: with it off, prompts longer than
  # max_num_batched_tokens are rejected outright.
  if "max_num_batched_tokens" not in sched_config_dict:
    sched_config.max_num_batched_tokens = 8192
  if "dynamic_split_fuse" not in sched_config_dict:
    sched_config.dynamic_split_fuse = True
  return generate_ov_scheduler_config(sched_config)


def check_vram_budget(loader: ModelLoadConfig) -> None:
  """Refuse a load that would oversubscribe VRAM.

     On the xe driver, allocations beyond VRAM do not fail: buffers spill into
     system RAM and the host can livelock until a watchdog reset (observed by
     Strata on an Arc Pro B70, 2026-09-29). Estimate weights (.bin bytes) +
     the configured KV pool (cache_size) + ~1.5 GB of headroom and compare
     against the device's total memory. When cache_size is unset OpenVINO
     sizes the KV pool dynamically from what is free, so only the weights +
     headroom floor is checked. OPENARC_VRAM_GUARD=0 disables the check.
  """
  device = (loader.device or "").upper()
  if not device.startswith("GPU"):
    return
  if os.getenv("OPENARC_VRAM_GUARD", "1") == "0":
    return
  import openvino as ov
  try:
    total = ov.Core().get_property(loader.device, "GPU_DEVICE_TOTAL_MEM_SIZE")
  except Exception as e:
    logger.warning(f"VRAM guard: could not query {loader.device} total memory ({e}); skipping check")
    return

  def _bin_bytes(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("*.bin"))

  weights = _bin_bytes(loader.model_path)
  if loader.draft_model_path:
    weights += _bin_bytes(loader.draft_model_path)
  sched = loader.scheduler_config
  kv_gb = sched.cache_size if (sched and sched.cache_size) else 0
  headroom = int(1.5e9)
  needed = weights + int(kv_gb * 1e9) + headroom
  if needed > total:
    raise RuntimeError(
      f"VRAM guard: '{loader.model_name}' needs at least ~{needed / 1e9:.1f} GB on "
      f"{loader.device} ({weights / 1e9:.1f} GB weights + {kv_gb} GB KV pool + "
      f"1.5 GB headroom) but the device has {total / 1e9:.1f} GB. On the xe driver, "
      f"oversubscribing VRAM spills into system RAM and can hang the host. Use a "
      f"smaller model/quantization, lower scheduler_config.cache_size, or set "
      f"OPENARC_VRAM_GUARD=0 to bypass this check."
    )
