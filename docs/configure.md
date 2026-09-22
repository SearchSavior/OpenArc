OpenArc now uses a YAML based configutation system! Before we did things with a CLI tool- but now, you are free to configure defaults to your hearts content. Configuration options set here override hardcoded defaults, but can be overridden at request time. For example, if you are in openwebui but need higher `max_tokens` than what you set in the config, you can raise that value in openwebui and that value will override the config saved in `config.yaml` for that request.


## Config Blocks

There are a few different blocks in a models config that accept different parameters. Some will be common across implementations while others are specific. 

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

OpenArc does not validate these, and OpenVINO upstream does not provide a way to check the behvaior of these settings in all cases. They can help you access hardware features not available to all devices like `SCHEDULING_CORE_TYPE` for more recent Intel CPUs, debug numeircal precision issues with `INFERENCE_PRECISION_HINT` or control `KV_CACHE_PRECISION`.

*properties* have the worst documentation in all of OpenVINO ecosystem, yet they are used everywhere in the openvino_notebooks, PRs and sometimes are even hardcoded depending on the needs of OpenVINO team. In that way, poking at these settings can drastically change performance but have less knobs than users of projects like `llama.cpp`, `vllm`, `sglang` are used to tinkering with.


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

### LLM

```yaml
models:
  qwen35-08b:
    load_config:
      engine: ovgenai
      model_type: llm
      model_path: /mnt/models/Qwen3.5-0.8B-int8-asym-ov
      device: CPU
      tool_call_parser: qwen35
    runtime_config:
      PERFORMANCE_HINT: LATENCY
    scheduler_config:
      max_num_batched_tokens: 2048
      num_kv_blocks: 
      cache_size:
      num_linear_attention_blocks:
      cache_interval_multiplier:
      dynamic_split_fuse:
      enable_prefix_caching:
      use_cache_eviction:
      use_sparse_attention:
    sampler_config:
      temperature: 0.7
      top_k: 40
      top_p: 0.95
      repetition_penalty: 1.05
      max_tokens: 1024
      chat_template_kwargs:
        enable_thinking: true
```

### VLM

```yaml
models:
  qwen35-08b:
    load_config:
      engine: ovgenai
      model_type: vlm
      model_path: /mnt/models/Qwen3.5-0.8B-int8-asym-ov
      device: CPU
      tool_call_parser: qwen35
    runtime_config:
      PERFORMANCE_HINT: LATENCY
    scheduler_config:
      max_num_batched_tokens:
      num_kv_blocks:
      cache_size:
      num_linear_attention_blocks:
      cache_interval_multiplier:
      dynamic_split_fuse:
      enable_prefix_caching:
      use_cache_eviction:
      use_sparse_attention:
    sampler_config:
      temperature: 0.7
      top_k: 40
      top_p: 0.95
      repetition_penalty: 1.05
      max_tokens: 1024
```

### Kokoro

OpenArc supports kokoro! Support in openvino has improved since I first implemented this model, so its possible we should look into review.

```yaml
models:
  kokoro-82m:
    load_config:
      engine: ovgenai
      model_type: kokoro
      model_path: /mnt/models/Kokoro-82M-ov
      device: CPU
    kokoro_config:
      voice: af_sarah                
      voice_blend: af_heart:0.7,af_nicole:0.3  
      lang_code: a
      speed: 1.0
      character_count_chunk: 400
      response_format: wav
```

### Qwen3-ASR

Another great model for ASR task. THe below settings around chunking have model-specific audio chunking logic which adapt and improve on what Qwen-Team released at launch- we study the energy of audio to choose when to slice up to `max_chunk_sec`, so you rarely hit `30` seconds. Therefore `max_chunk_sec` is a unit of inference, and performance should be judged by how long it takes to process a window.

Reference Implementation lives at [SearchSavior/Qwen3-ASR-OpenVINO](https://github.com/SearchSavior/Qwen3-ASR-OpenVINO)


```yaml
models:
  qwen3-asr:
    load_config:
      engine: ovgenai
      model_type: qwen3_asr
      model_path: /mnt/models/Qwen3-ASR-ov
      device: GPU.0
    qwen3_asr_config:
      language:                       # None = auto-detect
      max_tokens: 1024                # must be > 0
      max_chunk_sec: 30.0             # chunk upper bound, seconds
      search_expand_sec: 5.0          # boundary search expansion, seconds
      min_window_ms: 100.0            # energy window, ms
```

### Whisper

Whisper has no config block; audio arrives per request, so only load-time options apply.

```yaml
models:
  whisper-large-v3:
    load_config:
      engine: ovgenai
      model_type: whisper
      model_path: /mnt/models/whisper-large-v3-int8-ov
      device: GPU.0
    runtime_config:
      PERFORMANCE_HINT: LATENCY
```


### Qwen3-TTS

All 3 flavors of qwen3-tts are supported, with a rich set of configuration options. PRs to improve performance or other enhancements are welcome

Qwen3-TTS family models process audio and text in an interleaved way, leveraging the reasoning of the Qwen3-0.6b language model backbone to augment the latents produced up to that point; then, Qwen team trains a streaming audio codec decoder with multi-token prediction heads that spit audio into a tokenizer. 

If you want to learn more or contribute improvements, check out my reference implementation [SearchSavior/Qwen3-TTS-OpenVINO](https://github.com/SearchSavior/Qwen3-TTS-OpenVINO).



```yaml
models:
  qwen3-tts-custom:
    load_config:
      engine: ovgenai
      model_type: qwen3_tts_custom_voice
      model_path: /mnt/models/Qwen3-TTS-CustomVoice-ov
      device: GPU.0
    # Shared by all three modes.
    qwen3_tts_config:
      language:                       # None = auto-detect
      max_new_tokens: 2048
      do_sample: true
      top_k: 50
      top_p: 1.0
      temperature: 0.9
      repetition_penalty: 1.05
      non_streaming_mode: true        # false = drip-feed text during decode
      subtalker_do_sample: true
      subtalker_top_k: 50
      subtalker_top_p: 1.0
      subtalker_temperature: 0.9
      stream: true                    # chunked audio/L16 response
      stream_chunk_frames: 300
      stream_left_context: 25
```

#### Voice Clone

```yaml
    qwen3_tts_voice_clone_config:
      ref_text: "Transcript of the reference clip."  # enables ICL
      x_vector_only: false            # true = skip ICL even when ref_text is set
      instruct: "Speak slowly."
```

#### Voice Design
```yaml
    qwen3_tts_voice_design_config:
      voice_description: "A red furry muppet with an orange nose."
```
#### Custom Voice

```yaml
    qwen3_tts_custom_voice_config:
      speaker: chelsie
      instruct: "Sound cheerful."
```




















