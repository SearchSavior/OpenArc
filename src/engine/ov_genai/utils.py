"""OV GenAI engine utilities."""
import logging
from typing import Literal, Optional

from openvino_genai import SchedulerConfig

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import SchedulerConfigSchema
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

def generate_ov_scheduler_config(
    scheduler_config: SchedulerConfigSchema, context_window: Optional[int] = None
) -> dict:
  """Generates a SchedulerConfig object from the scheduler config model.

     `context_window` (the model's resolved context window, in tokens — either an
     explicit override via `openarc add --context-window` / the load config, or
     auto-discovered from config.json by ModelRegistry) becomes the compiled
     pipeline's *max content window*: it is written to
     `SchedulerConfig.max_num_batched_tokens`. That is not a cosmetic hint — in
     openvino.genai it is the knob that bounds a running sequence's KV-cache
     growth (pipeline_impl computes
     `max_sequence_cache_occupation_length_in_blocks =
       max_num_batched_tokens / block_size + 1`), and a sequence that cannot grow
     past it is flagged out-of-memory and its generation ends. So setting it is
     what actually makes a smaller window *bite* at inference time.

     Precedence for `max_num_batched_tokens`:
       1. an operator-set `max_num_batched_tokens` on the scheduler_config always wins;
       2. otherwise the resolved `context_window` (explicit override or config.json);
       3. otherwise openvino's built-in default (256).

     Note: a `scheduler_config` cannot be passed to SDPA pipelines without raising
     an error, so this helper is always entered through
     `extract_scheduler_config_from_loader`, which routes around the SDPA case.
  """
  sched_config = SchedulerConfig()

  # An operator-set max_num_batched_tokens wins; otherwise the resolved context
  # window becomes the pipeline's max content window. Both are non-positive-
  # aware: a truthy explicit value short-circuits before we consult context_window.
  max_num_batched_tokens = scheduler_config.max_num_batched_tokens
  if not max_num_batched_tokens and context_window:
    max_num_batched_tokens = context_window
  if max_num_batched_tokens:
    sched_config.max_num_batched_tokens = max_num_batched_tokens

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

def extract_scheduler_config_from_loader(
    loader: ModelLoadConfig,
) -> dict[Literal["scheduler_config"], SchedulerConfig]:
  """Extract the scheduler configuration from the loader and return it for the pipeline.

     A resolved `context_window` on the loader (set by ModelRegistry, either from
     an explicit `openarc add --context-window` / load-config override or
     auto-discovered from config.json) is applied here as the pipeline's max
     content window via `SchedulerConfig.max_num_batched_tokens`.

     A `scheduler_config` is emitted only when something meaningful is set — i.e.
     any non-None scheduler field, or a positive `context_window`. Pure-{"None": ...}
     scheduler blocks from the config template no longer force a phantom 256-token
     (openvino's built-in default) cap.

     If the pipeline uses the SDPA backend a `scheduler_config` is rejected by
     openvino.genai, so we return an empty dictionary: there the window can only be
     *advertised* (the record feeds /v1/models); the compiled model's own
     max_position_embeddings is what bounds its KV cache.
  """
  pipeline_kwargs = loader.runtime_config or {}
  sched_config = loader.scheduler_config or SchedulerConfigSchema()
  # exclude_none keeps purely-{"None": ...} template fields out of "is anything set",
  # so they no longer force a phantom max_num_batched_tokens==256 scheduler_config.
  sched_config_dict = sched_config.model_dump(exclude_unset=True, exclude_none=True)
  context_window = getattr(loader, "context_window", None)
  has_context_window = bool(context_window and context_window > 0)

  if pipeline_kwargs.get("ATTENTION_BACKEND") == "SDPA":
    if sched_config_dict:
      logger.error(
        "Cannot set scheduler_config for model: scheduler config is unsupported for SDPA backends"
      )
    elif has_context_window:
      logger.warning(
        "context_window=%s cannot be enforced at runtime: the model uses the SDPA backend, "
        "which does not accept a scheduler_config. It is still advertised in /v1/models; the "
        "compiled model's own max_position_embeddings bounds the KV cache instead.",
        context_window,
      )
    return {}
  if not sched_config_dict and not has_context_window:
    return {}
  return generate_ov_scheduler_config(sched_config, context_window)
