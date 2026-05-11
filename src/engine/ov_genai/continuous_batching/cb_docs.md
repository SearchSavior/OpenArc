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
