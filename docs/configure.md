OpenArc now uses a YAML based configutation system! Before we did things with a CLI tool- but now, you are free to configure defaults to your hearts content. 




# Examples

## LLM

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
      chat_template_kwargs:
        enable_thinking: true
```

## VLM

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

## Kokoro

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

## Qwen3-ASR

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

## Whisper

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


## Qwen3-TTS

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

### Voice Clone

```yaml
    qwen3_tts_voice_clone_config:
      ref_text: "Transcript of the reference clip."  # enables ICL
      x_vector_only: false            # true = skip ICL even when ref_text is set
      instruct: "Speak slowly."
```

### Voice Design
```yaml
    qwen3_tts_voice_design_config:
      voice_description: "A red furry muppet with an orange nose."
```
### Custom Voice

```yaml
    qwen3_tts_custom_voice_config:
      speaker: chelsie
      instruct: "Sound cheerful."
```

## scheduler_config

Scheduler properties (KV cache, prefix caching, batching) live on the [Performance](performance.md#scheduler_config) page.

## runtime_config

OpenVINO runtime *properties* and multi-device / speculative / caching recipes live on the [Performance](performance.md#runtime_config) page.