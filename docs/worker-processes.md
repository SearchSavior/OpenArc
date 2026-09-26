# Out-of-Process Inference Workers

OpenVINO models are not loaded in the server process. Each one runs in a
**dedicated worker subprocess** owned by a supervisor, and the server talks
to it over the process's stdin/stdout (one JSON object per line). The worker
builds and runs the pipeline; the server process never touches OpenVINO for
that model.

Two model families use two protocol dialects over the same byte-identical
framing (`src/engine/worker/protocol.py`, extended by
`src/engine/worker/plain/protocol.py`):

- **OpenVINO GenAI** (VLM, LLM, Whisper) — `OP_GENERATE` / `OP_TRANSCRIBE`
  stream tokens and segments. The process boundary exists because a wedged
  GPU plugin poisons the process-wide `ov::Core` and only a new process
  recovers (see below).
- **Plain OpenVINO** (Kokoro TTS, Qwen3-ASR, Qwen3-TTS) — `OP_RUN` (one
  result) and `OP_RUN_STREAM` (audio chunks, base64 float32 samples). The
  process boundary exists for **segfault isolation**: these engines crash
  natively, and a crash must take down only the worker, which the supervisor
  respawns — not the server.

A request line can be very large — a VLM chat request carries the whole
conversation (base64 images included) inside its JSON, so lines of tens of
MB are normal. Both sides therefore read the pipe with a line limit of
`PROTOCOL_LINE_LIMIT` (256 MiB, `protocol.py`), not asyncio's 64 KiB
`readline()` default; a line beyond the limit is reported as a clear `FATAL`
rather than killing the reader silently. The limit is per-model and can be
overridden with `worker_line_limit` in the model config (`openarc add
--worker-line-limit 256M`), in bytes or with a K/M/G suffix, minimum 64 KiB.
It bounds the IPC memory a single request can use; requests with larger
payloads fail with a clear error (the worker respawns). Because it is an IPC
setting rather than a compilation setting, changing it never invalidates the
compiled-model cache.

## Why a process boundary

openvino_genai pipelines share a **process-wide singleton `ov::Core`**. When
the GPU plugin wedges — `CL_OUT_OF_RESOURCES`, device loss, "could not execute
a primitive", `ProgramBuilder build failed`, ... — no in-process model unload
or recompile can fix it: the poisoned Core outlives the model. The only way to
get a clean Core is a new process. That is why:

| event | what happens |
| --- | --- |
| **unload** (`openarc unload`, API, or error-triggered) | the worker process is terminated. No wedged state can survive. |
| **load** (`openarc load`, startup) | a fresh process is spawned and the pipeline is built inside it (fresh Core). |
| **recoverable inference error** (the worker stays up) | the model is unloaded as before — now guaranteed clean, because unload is a process kill. |
| **non-recoverable error** (`CL_*` / driver failure) | the worker reports `FATAL` and exits; the supervisor **respawns a fresh process and re-runs the same load**, transparently. |
| **native crash / OOM-kill** (no `FATAL` message) | the supervisor detects the dead process and respawns it the same way. |
| **respawn budget exhausted** (2 respawns per load episode) | the model is unloaded from the registry: readiness drops, and an operator reloads it with a fresh budget. |

A PING watchdog (every 30 s, 5 s timeout) kills a worker that stops
responding, so a wedged-but-alive process is recovered the same way.

The plain-openvino engines (Kokoro, Qwen3-ASR, Qwen3-TTS) do not use
openvino_genai at all; they compile `ov::Model`s directly with `ov.Core`.
They do not wedge the Core — they **segfault** (native crashes inside the
OpenVINO/torch inference paths). A process boundary turns such a crash from
a server death into a worker death: the supervisor sees the dead process,
respawns, and the model comes back; the server process itself never crashes.
Audio crosses the pipe as base64 float32 sample arrays; WAV encoding stays
in the server, so the public response shapes are unchanged.

## What you will see in `openarc.log`

- `spawning inference worker: ...` / `inference worker ready (pid=...)`
- the worker's own stdout of OpenVINO/OpenCL (forwarded from its stderr),
  prefixed with `[<model> pid=...]`
- `inference worker exited unexpectedly (code=...)` + `respawning inference worker (n/2)` on recovery
- `respawn budget exhausted; worker is dead` + the resulting unload

## Operator knobs

| environment variable | effect |
| --- | --- |
| `OPENARC_OVGENAI_WORKER=0` | master switch: disable worker processes for all OpenVINO GenAI models (VLM/LLM/Whisper) and restore the historical in-process behaviour |
| `OPENARC_VLM_WORKER=0` | additionally disable the worker process for VLMs only (stage-1 escape hatch) |
| `OPENARC_OPENVINO_WORKER=0` | master switch: disable worker processes for the plain-openvino engines (Kokoro, Qwen3-ASR, Qwen3-TTS) and restore the historical in-process behaviour |

Everything else (respawn budget, watchdog timings, unload timeouts) is
configurable on `WorkerSupervisor` for now and will get config-file support in
a later stage.

## Implementation map

| file | role |
| --- | --- |
| `src/engine/worker/protocol.py` | wire protocol, error types, non-recoverable-error classification |
| `src/engine/worker/supervisor.py` | process lifecycle, protocol session, respawn + watchdog (inherited by both supervisors) |
| `src/engine/worker/worker_process.py` | GenAI child entry point; builds the pipeline inside the worker (base class for the plain worker) |
| `src/engine/worker/worker_client.py` | `RemoteEngine` base + `RemoteOVGenAI_*` facades (same surface as the engines) |
| `src/engine/worker/plain/protocol.py` | plain-OpenVINO dialect: re-exports the shared framing byte-identically, adds `OP_RUN` / `OP_RUN_STREAM` |
| `src/engine/worker/plain/supervisor.py` | `PlainWorkerSupervisor` — the same lifecycle, plain child entry point |
| `src/engine/worker/plain/worker_process.py` | plain child entry point; builds Kokoro / Qwen3-ASR / Qwen3-TTS in the worker |
| `src/engine/worker/plain/worker_client.py` | `RemoteOV_Kokoro` / `RemoteOVQwen3ASR` / `RemoteOVQwen3TTS` facades (same surface as the engines) |
| `src/server/model_registry.py` | the factory picks the facade by `(engine, model_type)` *before* instantiating (engine constructors have side effects) and wraps GenAI and plain-openvino engines |
| `src/server/worker_registry.py` | dispatches VLM/LLM/Whisper/Kokoro/ASR/TTS packets to the facades; a *worker death* does not trigger a registry unload (the supervisor owns recovery) |

## Scope (stages 1–3 done)

Stages 1–3 are complete: OpenVINO GenAI **VLM, LLM, and Whisper** and the
plain-openvino engines **Kokoro TTS, Qwen3-ASR, and Qwen3-TTS** (all three
voice modes) run out-of-process. The optimum engines (embedding, rerank)
still load in-process and join the worker pool in stage 4 — their call shape
returns tuples rather than streaming generators, which the single-result
`OP_RUN` op already supports. `openarc bench` also still builds its pipeline
in-process.
