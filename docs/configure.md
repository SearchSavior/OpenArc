OpenArc uses a YAML based configuration system! Models are added to `config.yaml` with `openarc add`, which writes the config blocks documented below. Configuration options set here override hardcoded defaults, but can be overridden at request time. For example, if you are in openwebui but need higher `max_tokens` than what you set in the config, you can raise that value in openwebui and that value will override the config saved in `config.yaml` for that request.


## Config Blocks

There are a few different blocks in a models config that accept different parameters. Some will be common across implementations while others are specific. `openarc add` writes to these blocks; `openarc add --help` shows one help panel per key.

### load_config
- engine: ovgenai, openvino, optimum
- model_type: llm, vlm, whisper, kokoro, emb, rerank, qwen3_tts_*
- device: device for this model
- tool_call_parser: gemma4, qwen35, hermes

### scheduler_config

These are knobs to directly control performance. you can read more in the documentation upstream [here](https://openvinotoolkit.github.io/openvino.genai/docs/category/optimization-techniques) 

- max_num_batched_tokens:
- num_kv_blocks:
- cache_size:
- num_linear_attention_blocks:
- cache_interval_multiplier:
- dynamic_split_fuse:
- enable_prefix_caching:
- use_cache_eviction:
- use_sparse_attention:

Openvino genai does a graph transformation in the pipeline, which adapts the model graph in place to choose a different codepath at runtime. The long load time on first load 

### runtime_config


runtime_config is an OpenArc entrypoint to the *properties* way of configuring openvino runtime. These settings allow users to tune the behavior of openivno runtime without needing to change application logic and are meant to be "portable", requring no code changes. Since OpenArc 

OpenArc does not validate these, and OpenVINO upstream does not provide a way to check the behvaior of these settings in all cases. They can help you access hardware features not available to all devices like `SCHEDULING_CORE_TYPE` for more recent Intel CPUs, debug numerical precision issues with `INFERENCE_PRECISION_HINT` or control `KV_CACHE_PRECISION`.

*properties* have the worst documentation in all of OpenVINO ecosystem, yet they are used everywhere in the openvino_notebooks, PRs and sometimes are even hardcoded depending on the needs of OpenVINO team. In that way, poking at these settings can drastically change performance but have less knobs than users of projects like `llama.cpp`, `vllm`, `sglang` are used to tinkering with. Additionally, some setting like `NUM_STREAMS` or ``


| Property | Values | Notes |
|---|---|---|
| `OFFLOAD_RATIO` | `0-100` | CPU offload |
| `ATTENTION_BACKEND` | `SDPA`, `PA` | |
| `KV_CACHE_PRECISION` | `u4`, `u8`, `f16`, `f32` | |
| `PERFORMANCE_HINT` | `LATENCY`, `THROUGHPUT` | |
| `EXECUTION_MODE_HINT` | `ACCURACY`, `PERFORMANCE` | |
| `INFERENCE_PRECISION_HINT` | `f16`, `f32` | |
| `MODEL_DISTRIBUTION_POLICY` | `TENSOR_PARALLEL`, `PIPELINE_PARALLEL` | |
| `ACTIVATIONS_SCALING_FACTOR` | | |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` | integer | |
| `ENABLE_HYPER_THREADING` | bool | defaults to true |
| `SCHEDULING_CORE_TYPE` | `ANY_CORE`, `ECORE_ONLY`, `PCORE_ONLY` | |
| `LOG_LEVEL` | `ERR`, `WARN`, `INFO`, `DEBUG`, `TRACE` | requires building openvino with DEBUG_CAPS=ON |


### sampler_config 

Request defaults for sampling, applied to `llm` and `vlm` models only. Anything set here can be overridden per request.

| Field | Default | Description |
|---|---|---|
| `temperature` | `1.0` | Sampling temperature; higher values increase randomness |
| `top_k` | `50` | Top-k sampling cutoff |
| `top_p` | `1.0` | Nucleus sampling probability cutoff |
| `repetition_penalty` | `1.0` | Penalty for repeating sequences of tokens |
| `frequency_penalty` | none | Penalty for repeated tokens |
| `presence_penalty` | none | Flat penalty for tokens which appeared at least once |
| `max_tokens` | `16384` | Maximum number of tokens to generate |
| `seed` | none | Fix the RNG seed; same prompt returns the same text |
| `chat_template_kwargs` | empty | Additional arguments to apply to the chat template (e.g. `enable_thinking: true`) |

## Example Configs

`openarc add` writes a model entry to `config.yaml`. Each help panel in `openarc add --help` is one config.yaml key: a flag is written to the key that backs it for the chosen `--model-type`, and a flag that does not apply to that model type is rejected. Only flags you actually pass are written, so everything else keeps its built-in default. Boolean options are plain flags (pass them to enable); options that take an object, like `--runtime-config` and `--chat-template-kwargs`, accept a JSON string.

### LLM

```
openarc add \
  --model-name qwen35-08b \
  --model-path /mnt/models/Qwen3.5-0.8B-int8-asym-ov \
  --engine ovgenai \
  --model-type llm \
  --device CPU \
  --tool-call-parser qwen35 \
  --runtime-config '{"PERFORMANCE_HINT": "LATENCY"}' \
  --max-num-batched-tokens 2048 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.95 \
  --repetition-penalty 1.05 \
  --max-tokens 1024 \
  --chat-template-kwargs '{"enable_thinking": true}'
```

### VLM

```
openarc add \
  --model-name qwen35-08b \
  --model-path /mnt/models/Qwen3.5-0.8B-int8-asym-ov \
  --engine ovgenai \
  --model-type vlm \
  --device CPU \
  --tool-call-parser qwen35 \
  --runtime-config '{"PERFORMANCE_HINT": "LATENCY"}' \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.95 \
  --repetition-penalty 1.05 \
  --max-tokens 1024
```

### Kokoro

OpenArc supports kokoro! Support in openvino has improved since I first implemented this model, so its possible we should look into review.

```
openarc add \
  --model-name kokoro-82m \
  --model-path /mnt/models/Kokoro-82M-ov \
  --engine ovgenai \
  --model-type kokoro \
  --device CPU \
  --voice af_sarah \
  --voice-blend af_heart:0.7,af_nicole:0.3 \
  --lang-code a \
  --speed 1.0 \
  --character-count-chunk 400 \
  --response-format wav
```

### Qwen3-ASR

Another great model for ASR task. THe below settings around chunking have model-specific audio chunking logic which adapt and improve on what Qwen-Team released at launch- we study the energy of audio to choose when to slice up to `max_chunk_sec`, so you rarely hit `30` seconds. Therefore `max_chunk_sec` is a unit of inference, and performance should be judged by how long it takes to process a window.

Reference Implementation lives at [SearchSavior/Qwen3-ASR-OpenVINO](https://github.com/SearchSavior/Qwen3-ASR-OpenVINO)

```
openarc add \
  --model-name qwen3-asr \
  --model-path /mnt/models/Qwen3-ASR-ov \
  --engine ovgenai \
  --model-type qwen3_asr \
  --device GPU.0 \
  --max-tokens 1024 \
  --max-chunk-sec 30.0 \
  --search-expand-sec 5.0 \
  --min-window-ms 100.0
```

### Whisper

Whisper has no config block; audio arrives per request, so only load-time options apply.

```
openarc add \
  --model-name whisper-large-v3 \
  --model-path /mnt/models/whisper-large-v3-int8-ov \
  --engine ovgenai \
  --model-type whisper \
  --device GPU.0 \
  --runtime-config '{"PERFORMANCE_HINT": "LATENCY"}'
```

### Qwen3-TTS

All 3 flavors of qwen3-tts are supported, with a rich set of configuration options. PRs to improve performance or other enhancements are welcome

Qwen3-TTS family models process audio and text in an interleaved way, leveraging the reasoning of the Qwen3-0.6b language model backbone to augment the latents produced up to that point; then, Qwen team trains a streaming audio codec decoder with multi-token prediction heads that spit audio into a tokenizer.

If you want to learn more or contribute improvements, check out my reference implementation [SearchSavior/Qwen3-TTS-OpenVINO](https://github.com/SearchSavior/Qwen3-TTS-OpenVINO).

#### Custom Voice

```
openarc add \
  --model-name qwen3-tts-custom \
  --model-path /mnt/models/Qwen3-TTS-CustomVoice-ov \
  --engine ovgenai \
  --model-type qwen3_tts_custom_voice \
  --device GPU.0 \
  --max-new-tokens 2048 \
  --do-sample \
  --top-k 50 \
  --top-p 1.0 \
  --temperature 0.9 \
  --repetition-penalty 1.05 \
  --non-streaming-mode \
  --subtalker-do-sample \
  --subtalker-top-k 50 \
  --subtalker-top-p 1.0 \
  --subtalker-temperature 0.9 \
  --stream \
  --stream-chunk-frames 300 \
  --stream-left-context 25 \
  --speaker chelsie \
  --instruct "Sound cheerful."
```

#### Voice Clone

Use `--model-type qwen3_tts_voice_clone` and add these flags to the shared set above:

```
  --ref-text "Transcript of the reference clip." \
  --instruct "Speak slowly."
```

Passing `--x-vector-only` skips ICL even when `--ref-text` is set.

#### Voice Design

Use `--model-type qwen3_tts_voice_design` and add:

```
  --voice-description "A red furry muppet with an orange nose."
```
