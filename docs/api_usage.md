---
icon: lucide/code
---

# API Usage

Request-time parameters are supplied per request via `extra_body` on the OpenAI-compatible API. Values set here override the model's config block in `config.yaml` (see [Configuration](configure.md)); anything omitted falls back to the config block, then the built-in defaults.

## Qwen3-TTS

Qwen3-TTS has three modes, selected by `model_type` in `config.yaml` (`qwen3_tts_custom_voice`, `qwen3_tts_voice_design`, `qwen3_tts_voice_clone`). Inference parameters (speaker, voice description, reference audio, sampling settings) are supplied per-request via the API.

CPU and GPU device are supported.

When GPU is selected as device, part of the model still runs on CPU.

Supported languages: `english`, `chinese`, `japanese`, `korean`, `german`, `french`, `spanish`, `italian`, `portuguese`, `russian`, `beijing_dialect`, `sichuan_dialect`. Pass `None` to auto-detect. See `demos/qwen3_tts_example.py` for a full request example.

### Custom Voice

Pick a predefined speaker at inference time (`serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`):

```python
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["OPENARC_API_KEY"],
)

response = client.audio.speech.create(
    model="<model-name>",
    input="Hello, this is a test.",
    extra_body={
        "openarc_tts": {
            "qwen3_tts": {
                # --- content ---
                "input": "Hello, this is a test.",
                "speaker": "uncle_fu",       # serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan
                "instruct": None,            # optional style instruction e.g. "Speak slowly and clearly."
                "language": "english",       # None to auto-detect
                # --- sampling ---
                "max_new_tokens": 2048,
                "do_sample": True,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "non_streaming_mode": True,
                "subtalker_do_sample": True,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
                "subtalker_temperature": 0.9,
                # --- streaming ---
                "stream": True,
                "stream_chunk_frames": 50,
                "stream_left_context": 25,
            }
        }
    },
)

Path("speech.wav").write_bytes(response.content)
```

### Voice Design

Describe the voice in free-form text at inference time:

```python
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["OPENARC_API_KEY"],
)

response = client.audio.speech.create(
    model="<model-name>",
    input="Hello, this is a test.",
    voice="alloy",
    extra_body={
        "openarc_tts": {
            "qwen3_tts": {
                # --- content ---
                "input": "Hello, this is a test.",
                "voice_description": "A calm, deep male voice with a slight British accent.",
                "language": "english",       # None to auto-detect
                # --- sampling ---
                "max_new_tokens": 2048,
                "do_sample": True,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "subtalker_do_sample": True,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
                "subtalker_temperature": 0.9,
                # --- streaming ---
                "stream": True,
                "stream_chunk_frames": 300,
                "stream_left_context": 25,
            }
        }
    },
)

Path("speech.wav").write_bytes(response.content)
```

### Voice Clone

Provide a reference WAV at inference time to clone a speaker:

```python
import base64
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["OPENARC_API_KEY"],
)

ref_audio_b64 = base64.b64encode(Path("reference.wav").read_bytes()).decode()

response = client.audio.speech.create(
    model="<model-name>",
    input="Hello, this is a test.",
    voice="alloy",
    extra_body={
        "openarc_tts": {
            "qwen3_tts": {
                # --- content ---
                "ref_audio_b64": ref_audio_b64,
                "ref_text": "Transcript of the reference audio.",  # optional, enables ICL
                "x_vector_only": False,      # True = x-vector only, skips ICL even if ref_text is set
                "instruct": None,            # optional style instruction
                "language": "english",       # None to auto-detect
                # --- sampling ---
                "max_new_tokens": 2048,
                "do_sample": True,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "subtalker_do_sample": True,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
                "subtalker_temperature": 0.9,
                # --- streaming ---
                "stream": True,
                "stream_chunk_frames": 300,
                "stream_left_context": 25,
            }
        }
    },
)

Path("speech.wav").write_bytes(response.content)
```

## Qwen3-ASR

Qwen3-ASR long-form transcription — supports Qwen3-ASR-0.6B. Audio is chunked automatically at silence boundaries up to `max_chunk_sec` (default `30s`). This is not a hard limit; chunking happens dynamically based on the energy of the audio.

Chunking can be configured on a per-request basis via `openarc_asr` in the request body for `/v1/audio/transcriptions`. Anything not set falls back to the model's `qwen3_asr_config` block in `config.yaml`, then the built-in defaults.

These `extra_body` options are OpenArc-specific, so third-party tools will not expose them; pass them yourself. Per-request tinkering works on both CPU and GPU. At this time NPU device is unsupported.

```python
import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["OPENARC_API_KEY"],
)

with Path("audio.wav").open("rb") as f:
    response = client.audio.transcriptions.create(
        model="<model-name>",
        file=f,
        response_format="verbose_json",
        # Optional. Values below are used as defaults if `openarc_asr` is not provided.
        extra_body={
            "openarc_asr": json.dumps({
                "qwen3_asr": {
                    "language": None,         # auto-detect, or e.g. "english"
                    "max_tokens": 1024,       # max tokens per chunk
                    "max_chunk_sec": 30.0,    # max audio chunk length in seconds
                    "search_expand_sec": 5.0, # silence-search window expansion
                    "min_window_ms": 100.0,   # minimum silence window in ms
                }
            })
        },
    )

print(response.text)
```
