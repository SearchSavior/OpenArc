# Qwen3-TTS Streaming Audio Output

## Overview

Add streaming audio output to the Qwen3-TTS codepath. During autoregressive generation, codec frames are periodically decoded into PCM audio chunks and streamed to the client via HTTP chunked transfer encoding. This uses the model's native streaming text-drip architecture (`non_streaming_mode=False`) combined with the upstream `chunked_decode` pattern.

## Motivation

Currently, Qwen3-TTS generates all codec frames, decodes the entire waveform, then returns a single WAV response. For long utterances this means high latency before any audio reaches the client. Streaming allows first audio to arrive as soon as the first chunk of frames is generated, reducing perceived latency significantly.

## Architecture

### Streaming Strategy: Mid-loop Chunked Decode

The autoregressive `_run_loop` generates codec frames one at a time (12Hz). Every 300 frames (~24s of audio), we decode the accumulated chunk through the speech decoder with 25 frames of left context from the previous chunk, trim the context audio, and yield the resulting PCM. These parameters match the upstream Qwen3-TTS `chunked_decode` specification.

This approach is used by every major open-source Qwen3-TTS serving implementation (groxaxo, faster-qwen3-tts, vLLM-Omni, etc.).

### Text-Drip Mode

When streaming, `non_streaming_mode` is set to `False`. This changes input construction: only the first text token enters the prefill, and remaining text tokens are drip-fed one per frame during the decode loop via the `trailing_text_hidden` mechanism already implemented in `_build_inputs`.

## Layer-by-Layer Design

### Layer 1: Engine (`src/engine/openvino/qwen3_tts/qwen3_tts.py`)

**New dataclass:**
```python
@dataclass
class TTSStreamChunk:
    audio: np.ndarray    # float32 PCM samples
    chunk_index: int
    is_final: bool
```

**New method: `_chunked_decode(codes, prev_codes)`**
- Takes current chunk of codec frames and up to 25 frames from the previous chunk
- Concatenates `[left_context, current_chunk]`, runs through speech decoder
- Trims audio samples corresponding to the left context frames
- Returns float32 PCM array

**New method: `_run_loop_streaming(inp, gen_config) -> Generator[TTSStreamChunk]`**
- Same autoregressive logic as `_run_loop`
- Accumulates frames in a buffer
- Every 300 frames: calls `_chunked_decode`, yields a `TTSStreamChunk`, resets buffer but retains last 25 frames as left context
- At EOS: decodes remaining frames (with left context) and yields final chunk with `is_final=True`

**New public method: `generate_stream(gen_config) -> AsyncIterator[TTSStreamChunk]`**
- Runs the sync streaming generator in a background thread
- Uses `asyncio.Queue` to bridge thread → async iterator
- Automatically sets `non_streaming_mode=False` on the gen config

**Existing methods unchanged:**
- `generate()` and `_run_loop` remain for non-streaming use

**Voice clone streaming:**
- `_generate_voice_clone_stream` variant handles reference prefix on first chunk only
- First chunk uses `_decode_icl`-style logic (prepend ref codes, decode, trim ref audio)
- Subsequent chunks use standard `_chunked_decode`

### Layer 2: Worker Registry (`src/server/worker_registry.py`)

**New method: `InferWorker.infer_qwen3_tts_stream(packet, tts_model) -> AsyncIterator[bytes]`**
- Calls `tts_model.generate_stream(packet.gen_config)`
- For each `TTSStreamChunk`: converts float32 audio to int16 PCM bytes (`np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()`)
- Yields raw bytes

**New method: `WorkerRegistry.generate_speech_qwen3_tts_stream(model_name, gen_config) -> AsyncIterator[bytes]`**
- Acquires the model from registry
- Delegates to `InferWorker.infer_qwen3_tts_stream`
- Handles error propagation

**Queue worker changes:**
- `queue_worker_qwen3_tts` detects streaming requests and routes to the streaming infer path
- Uses a `stream_queue` on the packet for async chunk delivery (same pattern as LLM streaming)

### Layer 3: API (`src/server/main.py`)

**Modified endpoint: `POST /v1/audio/speech`**
- When `request.stream is True` and model is Qwen3-TTS:
  - Returns `StreamingResponse(media_type="audio/pcm")`
  - Yields raw int16 PCM bytes from the streaming worker
- When `request.stream` is falsy: existing behavior unchanged

### Layer 4: Request Model (`src/server/models/requests_openai.py`)

**Add field to `OpenAISpeechRequest`:**
```python
stream: Optional[bool] = None
```

No changes to `OV_Qwen3TTSGenConfig` — chunk size (300) and left context (25) are engine-level constants matching upstream defaults.

## Audio Format Specification

| Property | Value |
|----------|-------|
| Encoding | Signed 16-bit integer (little-endian) |
| Sample rate | 24000 Hz |
| Channels | 1 (mono) |
| Byte order | Little-endian (native) |
| Chunk delivery | HTTP chunked transfer encoding |
| Content-Type | `audio/pcm` |
| End-of-stream | HTTP connection close |

## Client Requirements

A client consuming the streaming endpoint must:

1. Set `stream: true` and `response_format: "pcm"` in the request body
2. Read the response as a chunked byte stream (not JSON)
3. Know the audio format out-of-band: signed int16 LE, 24kHz, mono
4. Play or buffer chunks as they arrive — each chunk is contiguous int16 samples appendable to an audio playback buffer
5. Detect end-of-stream by the HTTP connection closing

### Example Client

```python
import httpx
import numpy as np

with httpx.stream("POST", "http://host:port/v1/audio/speech", json={
    "model": "custom_voice",
    "input": "Hello, this is a streaming test.",
    "stream": True,
    "response_format": "pcm",
    "openarc_tts": {
        "qwen3_tts": {
            "speaker": "Vivian",
            "language": "English",
        }
    }
}) as response:
    for chunk in response.iter_bytes():
        samples = np.frombuffer(chunk, dtype=np.int16)
        # feed samples to audio playback at 24kHz
```

## Constants

```python
STREAM_CHUNK_FRAMES = 300    # upstream recommended
STREAM_LEFT_CONTEXT = 25     # upstream recommended
SPEECH_DECODER_SR = 24000    # already defined
```

## Files Modified

| File | Change |
|------|--------|
| `src/engine/openvino/qwen3_tts/qwen3_tts.py` | Add `TTSStreamChunk`, `_chunked_decode`, `_run_loop_streaming`, `generate_stream`, voice clone streaming variant |
| `src/server/worker_registry.py` | Add streaming infer method, streaming worker routing, stream queue handling |
| `src/server/main.py` | Add streaming branch to `/v1/audio/speech` |
| `src/server/models/requests_openai.py` | Add `stream` field to `OpenAISpeechRequest` |

## Out of Scope

- WebSocket transport (HTTP chunked is sufficient and matches ecosystem)
- Configurable chunk size / left context (use upstream defaults; can add later if needed)
- MP3/Opus streaming encoding (PCM only for now)
- Crossfading at chunk boundaries (upstream left-context approach handles this)
