# ContinuousBatchingPipeline Docs

## Constructor Overload Example (L661 vs L664)

`ContinuousBatchingPipeline` has two useful Python init styles:

1. Internal tokenizer construction + explicit property maps (`L661`)
2. Injected tokenizer object + generic kwargs (`L664`)

```python
from __future__ import annotations

import openvino_genai as genai

MODEL_DIR = "/path/to/openvino/model_dir"
DEVICE = "GPU.0"

scheduler = genai.SchedulerConfig()
scheduler.max_num_batched_tokens = 2048
scheduler.max_num_seqs = 8
scheduler.cache_size = 4
scheduler.dynamic_split_fuse = True
scheduler.enable_prefix_caching = True

# L661-style: tokenizer is created internally from MODEL_DIR
pipe_internal_tok = genai.ContinuousBatchingPipeline(
    MODEL_DIR,
    scheduler_config=scheduler,
    device=DEVICE,
    properties={"PERFORMANCE_HINT": "THROUGHPUT"},
    tokenizer_properties={},
    vision_encoder_properties={},
)

# L664-style: pass a prebuilt tokenizer explicitly
tokenizer = genai.Tokenizer(MODEL_DIR)
pipe_injected_tok = genai.ContinuousBatchingPipeline(
    MODEL_DIR,
    tokenizer=tokenizer,
    scheduler_config=scheduler,
    device=DEVICE,
    PERFORMANCE_HINT="THROUGHPUT",
)

cfg = genai.GenerationConfig()
cfg.max_new_tokens = 64
cfg.do_sample = False

handle = pipe_internal_tok.add_request(1, "Explain continuous batching in one sentence.", cfg)
generated_ids: list[int] = []

while pipe_internal_tok.has_non_finished_requests():
    pipe_internal_tok.step()
    if handle.can_read():
        for output in handle.read().values():
            generated_ids.extend(output.generated_ids)

text = pipe_internal_tok.get_tokenizer().decode(generated_ids)
print(text)
```

### Practical Choice

- Use `L661` when you want OpenVINO GenAI to manage tokenizer construction from model assets.
- Use `L664` when you want direct control over tokenizer setup before pipeline creation.

## Installed Library API Surface

This section reflects the installed `openvino_genai` package inspected in the
OpenArc environment: `2026.2.0.0-3089-0a10767a25d`. Prefer this local package
surface over older online examples when implementing adapters.

### ContinuousBatchingPipeline

Constructor overloads:

```python
genai.ContinuousBatchingPipeline(
    models_path,
    scheduler_config,
    device,
    properties={},
    tokenizer_properties={},
    vision_encoder_properties={},
)

genai.ContinuousBatchingPipeline(
    models_path,
    tokenizer,
    scheduler_config,
    device,
    **kwargs,
)
```

OpenArc should use the first overload for the current adapter work:

```python
pipeline = genai.ContinuousBatchingPipeline(
    MODEL_DIR,
    scheduler_config=scheduler,
    device=DEVICE,
    properties=runtime_config,
    tokenizer_properties={},
    vision_encoder_properties={},
)
```

Request submission overloads:

```python
pipeline.add_request(request_id, input_ids: ov.Tensor, generation_config)
pipeline.add_request(request_id, prompt: str, generation_config)
pipeline.add_request(request_id, prompt: str, images: Sequence[ov.Tensor], generation_config)
pipeline.add_request(
    request_id,
    prompt: str,
    images: Sequence[ov.Tensor],
    videos: Sequence[ov.Tensor],
    generation_config,
    **kwargs,
)
```

Scheduler loop methods:

```python
pipeline.step()
pipeline.has_non_finished_requests()
pipeline.get_tokenizer()
pipeline.get_metrics()
pipeline.get_config()
```

### GenerationHandle

`add_request(...)` returns a `GenerationHandle`. The handle owns request-local
read and control state.

```python
handle.can_read() -> bool
handle.read() -> dict[int, genai.GenerationOutput]
handle.read_all() -> list[genai.GenerationOutput]
handle.get_status() -> genai.GenerationStatus
handle.cancel() -> None
handle.stop(finish_reason: genai.GenerationFinishReason = ...) -> None
```

`GenerationOutput` exposes:

```python
output.generated_ids -> list[int]
output.finish_reason -> genai.GenerationFinishReason
```

### GenerationStatus

Request handle status values:

```python
genai.GenerationStatus.RUNNING
genai.GenerationStatus.FINISHED
genai.GenerationStatus.IGNORED
genai.GenerationStatus.CANCEL
genai.GenerationStatus.STOP
```

Meaning:

- `RUNNING`: request is still active.
- `FINISHED`: request reached normal terminal completion.
- `IGNORED`: request ran into an out-of-memory condition and could not continue.
- `CANCEL`: request was cancelled; the last prompt and generated tokens are dropped from history.
- `STOP`: request was stopped; history keeps the prompt and generated tokens.

### GenerationFinishReason

Generation output finish reasons:

```python
genai.GenerationFinishReason.NONE
genai.GenerationFinishReason.STOP
genai.GenerationFinishReason.LENGTH
genai.GenerationFinishReason.TOOL_CALL
```

These are reported by `GenerationOutput.finish_reason`. A value other than
`NONE` indicates a terminal reason for that generation output.

### SchedulerConfig

Top-level continuous batching scheduler options:

```python
scheduler = genai.SchedulerConfig()
scheduler.max_num_batched_tokens = 2048
scheduler.max_num_seqs = 16
scheduler.cache_size = 8
scheduler.num_kv_blocks = 0
scheduler.dynamic_split_fuse = True
scheduler.enable_prefix_caching = True
scheduler.use_cache_eviction = False
scheduler.cache_eviction_config = eviction_config
scheduler.use_sparse_attention = False
scheduler.sparse_attention_config = sparse_attention_config
```

Available fields:

- `max_num_batched_tokens`: maximum total tokens scheduled in a batch.
- `max_num_seqs`: maximum scheduled sequences.
- `cache_size`: KV cache size in GB.
- `num_kv_blocks`: total KV blocks available to the scheduler.
- `dynamic_split_fuse`: split prompt and generation scheduling phases.
- `enable_prefix_caching`: keep prior KV blocks available for reuse.
- `use_cache_eviction`: enable token cache eviction during generation.
- `cache_eviction_config`: `CacheEvictionConfig`.
- `use_sparse_attention`: enable sparse attention during prefill.
- `sparse_attention_config`: `SparseAttentionConfig`.

The installed Python binding docstring mentions `block_size`, but it is not
exposed as an assignable Python property in the inspected package.

### CacheEvictionConfig

Constructor:

```python
eviction_config = genai.CacheEvictionConfig(
    start_size,
    recent_size,
    max_cache_size,
    aggregation_mode,
    apply_rotation=False,
    snapkv_window_size=8,
    kvcrush_config=None,
)
```

Parameters:

- `start_size`: tokens at the beginning of each sequence's KV cache to retain.
- `recent_size`: tokens at the end of each sequence's KV cache to retain.
- `max_cache_size`: maximum per-sequence tokens kept in KV cache.
- `aggregation_mode`: `AggregationMode`.
- `apply_rotation`: apply RoPE-based cache rotation after eviction.
- `snapkv_window_size`: window size for SnapKV-style importance score aggregation.
- `kvcrush_config`: optional KVCrush configuration from the lower-level binding.

Instance helpers:

```python
eviction_config.get_start_size()
eviction_config.get_recent_size()
eviction_config.get_max_cache_size()
eviction_config.get_evictable_size()
eviction_config.to_string()
```

### AggregationMode

Cache eviction aggregation modes:

```python
genai.AggregationMode.SUM
genai.AggregationMode.NORM_SUM
genai.AggregationMode.ADAPTIVE_RKV
```

Meaning:

- `SUM`: sum token importance scores after each generation step.
- `NORM_SUM`: sum scores normalized by token lifetime in cache.
- `ADAPTIVE_RKV`: use the Adaptive R-KV cache eviction algorithm.

### SparseAttentionConfig

Constructor:

```python
sparse_attention_config = genai.SparseAttentionConfig(
    mode=genai.SparseAttentionMode.TRISHAPE,
    num_last_dense_tokens_in_prefill=100,
    num_retained_start_tokens_in_cache=128,
    num_retained_recent_tokens_in_cache=1920,
    xattention_threshold=0.8,
    xattention_block_size=64,
    xattention_stride=8,
)
```

Parameters:

- `mode`: sparse attention mode.
- `num_last_dense_tokens_in_prefill`: final prompt tokens that still use dense attention.
- `num_retained_start_tokens_in_cache`: start-cache tokens retained for TRISHAPE.
- `num_retained_recent_tokens_in_cache`: recent-cache tokens retained for TRISHAPE.
- `xattention_threshold`: XAttention importance threshold.
- `xattention_block_size`: XAttention sparse block size.
- `xattention_stride`: XAttention importance-score sampling stride.

### SparseAttentionMode

Sparse attention modes:

```python
genai.SparseAttentionMode.TRISHAPE
genai.SparseAttentionMode.XATTENTION
```

Meaning:

- `TRISHAPE`: sparse prefill attention retaining configured start/recent cache regions.
- `XATTENTION`: block-sparse prefill attention using importance-score thresholding.

### PipelineMetrics

`pipeline.get_metrics()` returns process-level scheduler/cache counters. The
examples currently use:

```python
metrics.requests
metrics.scheduled_requests
metrics.cache_usage
metrics.max_cache_usage
metrics.avg_cache_usage
```

In the current handle-driven `add_request + step` path, per-request token timing
and throughput should be collected by OpenArc while draining handles.
