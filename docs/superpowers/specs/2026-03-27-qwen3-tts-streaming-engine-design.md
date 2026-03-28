# Qwen3-TTS Streaming Audio — Engine Layer

## Overview

Add streaming audio generation to `qwen3_tts.py` with real-time chunk yields via a new generator. Add details which map the codepath here.

## Motivation

The current pipeline generates all codec frames, decodes the full waveform, then returns it. For long utterances this means high latency before any audio is heard. Streaming decodes chunks mid-generation and plays them immediately, reducing first-audio latency to roughly `chunk_size / 12.5` seconds.

## Architecture

### Approach: Streaming `_run_loop`

A new `_run_loop_streaming()` generator runs the same autoregressive talker + code-predictor loop as `_run_loop`, but yields decoded audio chunks at configurable frame intervals instead of accumulating all frames. The existing non-streaming codepath is untouched.

### Chunked Decode Strategy

- Accumulate codec frames in a buffer
- At every `chunk_size` frames, concatenate up to `STREAM_LEFT_CONTEXT` frames from the previous chunk as left context
- Run through speech decoder, trim the context audio proportionally
- Yield the trimmed PCM


### Text-Drip Mode

Streaming forces `non_streaming_mode=False`. Only the first text token enters prefill; remaining tokens are drip-fed one per frame via the `trailing_text_hidden` mechanism already implemented in `_build_inputs`.

## Design



New fields on `OV_Qwen3TTSGenConfig`:

```python
stream_chunk_frames: int = 300
stream_left_context: int = 25
stream: bool = True
```

### Data Structure

```python
@dataclass
class TTSStreamChunk:
    audio: np.ndarray    # float32 PCM samples, 24kHz mono
    chunk_index: int
    is_final: bool
```

### `_chunked_decode(codes, prev_codes, perf) -> np.ndarray`

Decodes a chunk of codec frames with optional left context from the previous chunk.

- Converts `codes` (list of `[first_code] + subs` lists) to numpy, transposes to `(1, n_q, T)` via `arr.T[np.newaxis]` — same format as `_decode_codes`
- If `prev_codes` is provided, takes the last `STREAM_LEFT_CONTEXT` frames as context and prepends them
- If `prev_codes` is `None` (first chunk), decodes without left context
- Concatenates `[context, current_chunk]`, runs through speech decoder
- Trims output audio proportionally to remove context audio
- Returns float32 PCM array

### `_run_loop_streaming(inp, gen_config, perf) -> Generator[TTSStreamChunk]`

Same autoregressive logic as `_run_loop` with these differences:

- Accumulates frames in `all_codes` buffer, flushes at `chunk_size` boundary via `_chunked_decode`
- Carries `prev_codes` for left context across chunks
- Yields `TTSStreamChunk` at each chunk boundary
- Flushes remaining frames at EOS as the final chunk
- Full perf timing: talker prefill, code predictor prefill/decode splits, talker decode, per-frame averages, throughput — matching `_run_loop`. Speech decoder time is accumulated across chunk decodes in the shared `perf` dict; per-chunk decode time is also logged for tuning

### `generate_stream(gen_config) -> Generator[TTSStreamChunk]`

Synchronous public method (async wrapper deferred to serving-stack integration):

- Copies `gen_config` (via `model_copy(update={"non_streaming_mode": False})`) rather than mutating the caller's object
- Builds inputs using the same model-type dispatch as `_generate_standard` / `_generate_voice_clone` (supports all three model types: custom_voice, voice_design, voice_clone)
- Delegates to `_run_loop_streaming`, yielding chunks directly
- Speech decoder perf is accumulated across chunks in the shared `perf` dict; per-chunk decode time is logged for tuning

### `__main__` Entrypoint

Adds CLI flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stream` | bool | `False` | Enable streaming playback |
| `--chunk-frames` | int | `50` | Codec frames per chunk |

When `--stream`:
- Opens a `sounddevice.OutputStream(samplerate=24000, channels=1, dtype="float32")`
- Iterates `generate_stream()` (sync generator), writes each chunk's audio to the stream
- `stream.write()` blocks if device buffer is full (natural backpressure)
- Logs each chunk: index, sample count, duration, final marker

When not `--stream`: existing behavior (voice-clone WAV file output).

Note: the `__main__` entrypoint is currently hardcoded for voice-clone mode. `--stream` / `--chunk-frames` are additions to this existing test script. Generalizing to all three model types is out of scope.

## Files Modified

| File | Change |
|------|--------|
| `src/engine/openvino/qwen3_tts/qwen3_tts.py` | Add `TTSStreamChunk`, `_chunked_decode`, `_run_loop_streaming`, `generate_stream`; update `__main__` with `--stream` / `--chunk-frames` |
| `src/server/models/openvino.py` | Add `stream_chunk_frames: int = 50` to `OV_Qwen3TTSGenConfig` |

## Dependencies

- `sounddevice` — only imported in `__main__` when `--stream` is used, not a server dependency

## Codebase Integration Plan

Once streaming is validated via the `__main__` entrypoint, integrate into the serving stack. The streaming codepath is kept **entirely separate** from the existing non-streaming path — controlled by the `stream`, `stream_chunk_frames`, and `stream_left_context` fields already on `OV_Qwen3TTSGenConfig` (lines 40–42 of `openvino.py`), which live under the `openarc_tts.qwen3_tts` namespace in the request body.

### Layer 1: Request Model

No changes to `OpenAISpeechRequest`. The `stream` flag lives on `OV_Qwen3TTSGenConfig` (under `openarc_tts.qwen3_tts`), not at the top level — streaming is Qwen3-TTS-specific (Kokoro has no autoregressive loop to stream from). The API layer reads `gen_config.stream` to decide the HTTP response mode.

### Layer 2: Worker Registry (`src/server/worker_registry.py`)

**New static method: `InferWorker.infer_qwen3_tts_stream`**
```python
@staticmethod
async def infer_qwen3_tts_stream(packet: WorkerPacket, tts_model: OVQwen3TTS) -> None:
```
- Calls `tts_model.generate_stream(packet.gen_config)` (sync generator)
- Runs in a thread via `asyncio.to_thread` or equivalent
- For each `TTSStreamChunk`: converts float32 audio to int16 PCM bytes, puts onto `packet.stream_queue`
- Puts `None` sentinel when generator exhausts
- Error handling: puts `None` on exception so consumer doesn't hang

**New method: `WorkerRegistry.stream_generate_speech_qwen3_tts`**
```python
async def stream_generate_speech_qwen3_tts(
    self, model_name: str, gen_config: OV_Qwen3TTSGenConfig
) -> AsyncIterator[bytes]:
```
- Follows the exact pattern of `WorkerRegistry.stream_generate` (lines 801–831):
  - Creates `stream_queue`, `result_future`, `WorkerPacket`
  - Puts packet on the qwen3_tts model queue
  - Yields items from `stream_queue` until `None` sentinel
- Audio format: raw int16 LE PCM bytes per chunk

**Modify `QueueWorker.queue_worker_qwen3_tts`:**
- Check `packet.gen_config.stream`:
  - If `True` and `packet.stream_queue is not None`: route to `InferWorker.infer_qwen3_tts_stream`
  - Otherwise: existing `InferWorker.infer_qwen3_tts` path (unchanged)

### Layer 3: API Endpoint (`src/server/main.py`)

**Modify `POST /v1/audio/speech` handler (`openai_audio_speech`):**

After the existing Qwen3 TTS branch (line ~719), add streaming dispatch based on `gen_config.stream`:

```python
if normalized in (QWEN3_TTS_CUSTOM_VOICE, QWEN3_TTS_VOICE_DESIGN, QWEN3_TTS_VOICE_CLONE):
    if not request.openarc_tts or not request.openarc_tts.qwen3_tts:
        raise ValueError("openarc_tts.qwen3_tts required for Qwen3 TTS models")
    gen_config = request.openarc_tts.qwen3_tts
    gen_config.input = request.input

    if gen_config.stream:
        return StreamingResponse(
            _workers.stream_generate_speech_qwen3_tts(request.model, gen_config),
            media_type="audio/pcm",
        )

    result = await _workers.generate_speech_qwen3_tts(request.model, gen_config)
```

Non-streaming path is completely unchanged.

### Layer 4: Audio Format Specification

| Property | Streaming | Non-streaming |
|----------|-----------|---------------|
| Encoding | int16 LE PCM | WAV (existing) |
| Sample rate | 24000 Hz | 24000 Hz |
| Channels | 1 (mono) | 1 (mono) |
| Content-Type | `audio/pcm` | `audio/wav` |
| Delivery | HTTP chunked transfer | Single response |
| End-of-stream | HTTP connection close | Content-Length |

### Data Flow: Streaming

```
Client POST /v1/audio/speech
  { "openarc_tts": { "qwen3_tts": { "stream": true, "stream_chunk_frames": 50 } } }
    ↓
main.py: reads gen_config.stream == True
    ↓
WorkerRegistry.stream_generate_speech_qwen3_tts()
  → creates stream_queue, puts WorkerPacket on qwen3_tts queue
    ↓
queue_worker_qwen3_tts: detects stream=True
  → calls InferWorker.infer_qwen3_tts_stream()
    ↓
OVQwen3TTS.generate_stream(gen_config)
  → _run_loop_streaming yields TTSStreamChunk
    ↓
infer_qwen3_tts_stream: float32→int16 PCM bytes → stream_queue.put()
    ↓
stream_generate_speech_qwen3_tts: yields bytes from stream_queue
    ↓
StreamingResponse(media_type="audio/pcm")
    ↓
Client receives chunked int16 PCM at 24kHz
```

### Client Example

```python
import httpx
import numpy as np

with httpx.stream("POST", "http://host:port/v1/audio/speech", json={
    "model": "custom_voice",
    "input": "Hello, this is a streaming test.",
    "openarc_tts": {
        "qwen3_tts": {
            "speaker": "Vivian",
            "language": "English",
            "stream": True,
            "stream_chunk_frames": 50,
        }
    }
}) as response:
    for chunk in response.iter_bytes():
        samples = np.frombuffer(chunk, dtype=np.int16)
        # feed samples to audio playback at 24kHz
```

### Files Modified (Integration Phase)

| File | Change |
|------|--------|
| `src/server/worker_registry.py` | Add `infer_qwen3_tts_stream`, `stream_generate_speech_qwen3_tts`; modify `queue_worker_qwen3_tts` for stream routing |
| `src/server/main.py` | Add streaming branch to `/v1/audio/speech` handler |

No changes to `OV_Qwen3TTSGenConfig` — the `stream`, `stream_chunk_frames`, and `stream_left_context` fields are already there from the engine layer work.

## Out of Scope

- Voice clone ICL first-chunk handling (use standard `_chunked_decode` for now; ICL-aware first chunk can be added if quality issues arise)
- Crossfading at chunk boundaries (left context approach handles continuity)
- MP3/Opus encoding
- WebSocket transport
