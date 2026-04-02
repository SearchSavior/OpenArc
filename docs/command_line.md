# CLI Reference


Add a model to `openarc_config.json` for easy loading with `openarc load`. 


### Single device

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine <engine> \
  --model-type <model-type> \
  --device <target-device>
```

To see what options you have for `--device`, use `openarc tool device-detect`.

### VLM

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine <engine> \
  --model-type <model-type> \
  --device <target-device> \
  --vlm-type <vlm-type>
```
Getting VLM to work the way I wanted required using VLMPipeline in ways that are not well documented. You can look at the [code](src/engine/ov_genai/vlm.py#L33) to see where the magic happens. 

`vlm-type` maps a vision token for a given architecture using strings like `qwen25vl`, `phi4mm` and more. Use `openarc add --help` to see the available options. The server will complain if you get anything wrong, so it should be easy to figure out.

### Whisper

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/whisper> \
  --engine ovgenai \
  --model-type whisper \
  --device <target-device>
```

### Kokoro (CPU only)

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/kokoro> \
  --engine openvino \
  --model-type kokoro \
  --device CPU
```

### Advanced Configuration Options

`runtime-config` accepts many options to modify `openvino` runtime behavior for different inference scenarios. OpenArc reports c++ errors to the server when these fail, making experimentation easy. 

See OpenVINO documentation on [Inference Optimization](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference.html) to learn more about what can be customized.

Most options get really deep into OpenVINO concepts that are way out of scope for the README; however `runtime-config` is the entrypoint for *all* of them. Broadly, what you set in `runtime-config` Unfortunately, not all options are designed for transformers, so `runtime-config` was implemented in a way where you immediately get feedback. Add a kwarg, load the model, get feedback from the server, run `openarc bench`. Overall, it's a clean way to handle the hardest part of OpenVINO documentation. 
 
Review [pipeline-paralellism preview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how you can customize multi device inference using HETERO device plugin. Some example commands are provided for a few difference scenarios:

### Multi-GPU Pipeline Paralell

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine ovgenai \
  --model-type llm \
  --device HETERO:GPU.0,GPU.1 \
  --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}"
```

### Tensor Paralell (CPU only)

Requires more than one CPU socket in a single node.

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine ovgenai \
  --model-type llm \
  --device CPU \
  --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}"
```
---

### Hybrid Mode/CPU Offload

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine ovgenai \
  --model-type llm \
  --device HETERO:GPU.0,CPU \
  --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}"
```

### Speculative Decoding

```
openarc add \
  --model-name <model-name> \
  --model-path <path/to/model> \
  --engine ovgenai \
  --model-type llm \
  --device GPU.0 \
  --draft-model-path <path/to/draftmodel> \
  --draft-device CPU \
  --num-assistant-tokens 5 \
  --assistant-confidence-threshold 0.5
```