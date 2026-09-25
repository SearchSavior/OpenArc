---
icon: lucide/gauge
---

# Performance

Tuning knobs set as flags on `openarc add`, stored in a model's entry in `config.yaml` alongside `load_config`. See [Configuration](configure.md) for the full config layout and per-model-type blocks.

## scheduler_config

OpenVINO GenAI scheduler properties for `llm` and `vlm` models. Not all fields apply in all cases, see the upstream [OpenVINO GenAI Docs](https://openvinotoolkit.github.io/openvino.genai/docs/category/optimization-techniques) for more details

```
--max-num-batched-tokens <int>      # max tokens per batch (across all sequences)
--max-num-seqs <int>                # max scheduled sequences ("max batch size")
--num-kv-blocks <int>               # total KV blocks available to the scheduler
--cache-size <int>                  # total cache size in GB
--num-linear-attention-blocks <int> # linear attention models only
--cache-interval-multiplier <int>   # linear-attn checkpoint interval multiplier (default 8)
--dynamic-split-fuse                # split prompt / generate into separate scheduling phases
--enable-prefix-caching             # keep KV blocks for reuse across sequences
--use-cache-eviction                # evict cache during generation
--use-sparse-attention              # sparse attention during prefill
```

## Multi-Device Inference

Review the [pipeline-parallelism preview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how to customize multi-device inference using the HETERO device plugin.

### CPU Offload for MoE Models

So far we have tested Qwen3.6-35B-A3B on B580 and B70 and it works, but this is a preview feature upstream so mileage may vary.

```
--runtime-config '{"OFFLOAD_RATIO": 1}'
```

### Multi-GPU Pipeline Parallel
Note: this feature has issues and may not work. 

```
--device HETERO:GPU.0,GPU.1 \
--runtime-config '{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}'
```

### Tensor Parallel

Requires more than one CPU socket in a single node.

```
--device CPU \
--runtime-config '{"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}'
```

### Hybrid / CPU Offload

```
--device HETERO:GPU.0,CPU \
--runtime-config '{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}'
```

## Speculative Decoding

Draft-model speculative decoding. The version configured here uses a smaller draft model which shares a tokenizer; we still are validating the upstream implementation of MTP and Dflash but no code changes should be neccessary for them to work.

```
openarc add \
  --model-name qwen35-08b \
  --model-path /mnt/models/Qwen3.5-0.8B-int8-asym-ov \
  --engine ovgenai \
  --model-type llm \
  --device GPU.0 \
  --draft-model-path /mnt/models/Qwen3.5-0.5B-int8-asym-ov \
  --draft-device CPU \
  --num-assistant-tokens 5 \
  --assistant-confidence-threshold 0.5
```

## Model Caching

`cache_dir` caches compiled model blobs on first load, greatly reducing startup time and memory cost for subsequent process starts. On some setups this can reduce peak memory utilization on restarts by 3x or more, and start time by 7x. For additional details, see the [OpenVINO model cache docs](https://docs.openvino.ai/2026/model-server/ovms_docs_model_cache.html).

```
--device GPU \
--cache-dir /model_path/model_cache
```

The cache can be shared by multiple processes (and on shared network filesystems such as NFS or CephFS) provided that only one process updates it at a time.

The cache will be fully or partially invalidated when doing any of the below:

- Changing the utilized device(s) (swapping GPU models, adding or removing a GPU, adding or removing a CPU, etc.)
- Changing `runtime_config` that impacts the model itself (e.g. `PERFORMANCE_HINT: THROUGHPUT` to `PERFORMANCE_HINT: LATENCY`, but not `NUM_STREAMS: 1` to `NUM_STREAMS: 2`)
- Changing any part of the software stack from the firmware up — GPU firmware, OS kernel, kernel modules/drivers, dependency libraries, OpenArc, model versions

> **Warning:** Due to OpenVINO limitations, unused cache files are never cleaned up and will persist until an operator removes them. The cache can grow large over time. It is recommended that operators monitor the cache size and manually clean it up as needed to reduce disk usage.
