# Out-of-Process Inference Workers

OpenVINO GenAI VLMs are not loaded in the server process. Each one runs in a
**dedicated worker subprocess** owned by a supervisor, and the server talks to
it over the process's stdin/stdout (one JSON object per line). The worker
builds and runs the `VLMPipeline`; the server process never touches OpenVINO
for that model.

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

## What you will see in `openarc.log`

- `spawning inference worker: ...` / `inference worker ready (pid=...)`
- the worker's own stdout of OpenVINO/OpenCL (forwarded from its stderr),
  prefixed with `[<model> pid=...]`
- `inference worker exited unexpectedly (code=...)` + `respawning inference worker (n/2)` on recovery
- `respawn budget exhausted; worker is dead` + the resulting unload

## Operator knobs

| environment variable | effect |
| --- | --- |
| `OPENARC_VLM_WORKER=0` | disable the worker process for VLMs and restore the historical in-process behaviour (escape hatch for debugging) |

Everything else (respawn budget, watchdog timings, unload timeouts) is
configurable on `WorkerSupervisor` for now and will get config-file support in
a later stage.

## Implementation map

| file | role |
| --- | --- |
| `src/engine/worker/protocol.py` | wire protocol, error types, non-recoverable-error classification |
| `src/engine/worker/supervisor.py` | process lifecycle, protocol session, respawn + watchdog |
| `src/engine/worker/worker_process.py` | child entry point; builds the pipeline inside the worker |
| `src/engine/worker/worker_client.py` | `RemoteOVGenAI_VLM` facade (same surface as `OVGenAI_VLM`) |
| `src/server/model_registry.py` | the factory wraps `OVGenAI_VLM` in the facade |
| `src/server/worker_registry.py` | dispatches VLM packets to the facade; a *worker death* does not trigger a registry unload (the supervisor owns recovery) |

## Scope (stage 1)

This is stage 1 of a larger refactor: only the OpenVINO GenAI **VLM** engine
runs out-of-process. LLM, Whisper, and the plain-openvino engines (Kokoro,
Qwen3-ASR/TTS) still load in-process and will join the worker pool in later
stages. `openarc bench` also still builds its pipeline in-process.
