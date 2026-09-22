---
icon: lucide/gauge
---

# Performance

Tuning knobs which live in a model's entry in `config.yaml`, alongside `load_config`. See [Configuration](configure.md) for the full config layout and per-model-type blocks.

## scheduler_config

OpenVINO GenAI scheduler properties for `llm` and `vlm` models. All fields are optional; unset fields use engine defaults.

```yaml
models:
  qwen35-08b:
    load_config:
      engine: ovgenai
      model_type: llm
      model_path: /mnt/models/Qwen3.5-0.8B-int8-asym-ov
      device: CPU
    scheduler_config:
      max_num_batched_tokens:    # max tokens per batch (across all sequences)
      max_num_seqs:              # max scheduled sequences ("max batch size")
      num_kv_blocks:             # total KV blocks available to the scheduler
      cache_size:                # total cache size in GB
      num_linear_attention_blocks:  # linear attention models only
      cache_interval_multiplier: # linear-attn checkpoint interval multiplier (default 8)
      dynamic_split_fuse:        # split prompt / generate into separate scheduling phases
      enable_prefix_caching:     # keep KV blocks for reuse across sequences
      use_cache_eviction:        # evict cache during generation
      use_sparse_attention:      # sparse attention during prefill
```

Notes from the schema:

- `num_linear_attention_blocks` and `cache_interval_multiplier` only apply to models with linear attention cache inputs. `cache_interval_multiplier: 0` is valid only when prefix caching is disabled.
- With `enable_prefix_caching` on, all previously calculated KV caches are kept in memory and blocks are not released; maximum RAM usage is bounded by `cache_size` or `num_kv_blocks`. With it off, only the KV cache required for the current batch is kept and released when a sequence finishes.


## Multi-Device Inference

Review the [pipeline-parallelism preview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how to customize multi-device inference using the HETERO device plugin.

### CPU Offload for MoE Models

So far we have tested Qwen3.6-35B-A3B on B580 and B70 and it works, but this is a preview feature upstream.

```yaml
runtime_config
  OFFLOAD_RATIO: 1
```

### Multi-GPU Pipeline Parallel
Note: this feature has issues and may not work. 

```yaml
    load_config:
      device: HETERO:GPU.0,GPU.1
    runtime_config:
      MODEL_DISTRIBUTION_POLICY: PIPELINE_PARALLEL
```

### Tensor Parallel

Requires more than one CPU socket in a single node.

```yaml
    load_config:
      device: CPU
    runtime_config:
      MODEL_DISTRIBUTION_POLICY: TENSOR_PARALLEL
```

### Hybrid / CPU Offload

```yaml
    load_config:
      device: HETERO:GPU.0,CPU
    runtime_config:
      MODEL_DISTRIBUTION_POLICY: PIPELINE_PARALLEL
```

## Speculative Decoding

Draft-model speculative decoding, enabled per model entry. Enables a 1.3-1.4x speedup.

```yaml
models:
  qwen35-08b:
    load_config:
      engine: ovgenai
      model_type: llm
      model_path: /mnt/models/Qwen3.5-0.8B-int8-asym-ov
      device: GPU.0
      draft_model_path: /mnt/models/Qwen3.5-0.5B-int8-asym-ov
      draft_device: CPU
      num_assistant_tokens: 5
      assistant_confidence_threshold: 0.5
```

## Model Caching

`cache_dir` caches compiled model blobs on first load, greatly reducing startup time and memory cost for subsequent process starts. On some setups this can reduce peak memory utilization on restarts by 3x or more, and start time by 7x. For additional details, see the [OpenVINO model cache docs](https://docs.openvino.ai/2026/model-server/ovms_docs_model_cache.html).

```yaml
    load_config:
      device: GPU
      cache_dir: /mnt/models/cache/qwen35-08b
```

The cache can be shared by multiple processes (and on shared network filesystems such as NFS or CephFS) provided that only one process updates it at a time.

The cache will be fully or partially invalidated when doing any of the below:

- Changing the utilized device(s) (swapping GPU models, adding or removing a GPU, adding or removing a CPU, etc.)
- Changing `runtime_config` that impacts the model itself (e.g. `PERFORMANCE_HINT: THROUGHPUT` to `PERFORMANCE_HINT: LATENCY`, but not `NUM_STREAMS: 1` to `NUM_STREAMS: 2`)
- Changing any part of the software stack from the firmware up — GPU firmware, OS kernel, kernel modules/drivers, dependency libraries, OpenArc, model versions

> **Warning:** Due to OpenVINO limitations, unused cache files are never cleaned up and will persist until an operator removes them. The cache can grow large over time. It is recommended that operators monitor the cache size and manually clean it up as needed to reduce disk usage.
