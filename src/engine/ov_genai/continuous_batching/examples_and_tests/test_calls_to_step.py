"""
Continuous-batching behavior probe — OpenVINO GenAI 2026.2

Run this on the machine with the Intel GPU:

    python cb_probe.py

It does NOT touch the OpenArc server. It isolates the two questions that
decide how the CB engine loop should be threaded.

  EXPERIMENT 1 - Does ContinuousBatchingPipeline.step() release the GIL?
      A monitor thread runs a tight pure-Python loop and counts how many
      iterations it completes per second. We measure that rate with the
      pipeline idle (baseline), then again while the main thread drives
      step() in a loop. If step() releases the GIL during the forward
      pass, the monitor keeps running near baseline rate. If step() holds
      the GIL, the monitor is starved and the rate collapses.

  EXPERIMENT 2 - Does an asyncio loop stay responsive when step() is driven
      through a single-worker ThreadPoolExecutor?
      An engine-loop coroutine runs step() via loop.run_in_executor() on a
      max_workers=1 pool. A heartbeat coroutine that wants to wake every
      10 ms records its jitter. Low jitter => the recommended architecture
      keeps the event loop free. A third coroutine injects new requests
      mid-generation to confirm add_request() works on a live pipeline.

Tune the CONFIG block for your hardware. Verdicts print at the end.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import openvino_genai as genai


# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #

MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Anubis-Mini-8B-v1-int4_asym-ov/"
DEVICE = "GPU.0"

# SchedulerConfig - tune cache_size (GB) to fit your VRAM.
SCHED_MAX_NUM_BATCHED_TOKENS = 2048
SCHED_MAX_NUM_SEQS = 16
SCHED_CACHE_SIZE_GB = 8
SCHED_DYNAMIC_SPLIT_FUSE = True
SCHED_ENABLE_PREFIX_CACHING = True

MAX_NEW_TOKENS = 192          # enough steps to measure, short enough to stay quick
BASELINE_SECONDS = 3.0        # how long to measure the idle monitor baseline
HEARTBEAT_TARGET_MS = 10.0    # experiment 2 heartbeat period

PROMPTS = [
    "Explain how continuous batching differs from static batching for LLM serving.",
    "Write a short paragraph about the history of the printing press.",
    "Describe, step by step, how a CPU executes a single instruction.",
    "Summarize the causes of the 2008 financial crisis in plain language.",
]

# Injected mid-run in experiment 2 to test add_request() on a live pipeline.
INJECT_PROMPTS = [
    "List five considerations when choosing a database index.",
    "Explain what a KV cache is and why it matters for transformer inference.",
]

TERMINAL_STATUSES = {
    genai.GenerationStatus.FINISHED,
    genai.GenerationStatus.IGNORED,
    genai.GenerationStatus.CANCEL,
    genai.GenerationStatus.STOP,
}


# --------------------------------------------------------------------------- #
# Pipeline setup                                                              #
# --------------------------------------------------------------------------- #

def build_pipeline() -> genai.ContinuousBatchingPipeline:
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = SCHED_MAX_NUM_BATCHED_TOKENS
    scheduler.max_num_seqs = SCHED_MAX_NUM_SEQS
    scheduler.cache_size = SCHED_CACHE_SIZE_GB
    scheduler.dynamic_split_fuse = SCHED_DYNAMIC_SPLIT_FUSE
    scheduler.enable_prefix_caching = SCHED_ENABLE_PREFIX_CACHING

    print(f"[setup] loading {MODEL_PATH} on {DEVICE} ...")
    t0 = time.perf_counter()
    pipe = genai.ContinuousBatchingPipeline(
        MODEL_PATH,
        device=DEVICE,
        scheduler_config=scheduler,
    )
    print(f"[setup] pipeline ready in {time.perf_counter() - t0:.1f}s")
    return pipe


def build_generation_config() -> genai.GenerationConfig:
    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    cfg.do_sample = False
    return cfg


def drain_handles(
    handles: dict[int, genai.GenerationHandle],
    token_counts: dict[int, int],
) -> list[int]:
    """Read newly generated tokens from each handle; return finished ids.

    Note the terminal check covers IGNORED/CANCEL/STOP, not just FINISHED -
    an OOM-IGNORED request must be retired too, or the loop never drops it.
    """
    finished: list[int] = []
    for rid, handle in handles.items():
        if handle.can_read():
            for output in handle.read().values():
                token_counts[rid] = token_counts.get(rid, 0) + len(output.generated_ids)
        if handle.get_status() in TERMINAL_STATUSES:
            finished.append(rid)
    return finished


# --------------------------------------------------------------------------- #
# Experiment 1 - GIL release probe                                            #
# --------------------------------------------------------------------------- #

@dataclass
class MonitorResult:
    iterations: int = 0
    seconds: float = 0.0
    max_stall_ms: float = 0.0
    stalls_over_1ms: int = 0

    @property
    def iters_per_sec(self) -> float:
        return self.iterations / self.seconds if self.seconds > 0 else 0.0


def _run_monitor(stop: threading.Event, result: MonitorResult) -> None:
    """Tight pure-Python loop. It only fails to make progress when it
    cannot acquire the GIL, so its iteration rate is a direct proxy for
    'how much of the time was the GIL available to other threads'."""
    iterations = 0
    max_stall = 0.0
    stalls = 0
    start = time.perf_counter()
    prev = start
    while not stop.is_set():
        now = time.perf_counter()
        delta = now - prev
        prev = now
        iterations += 1
        if delta > max_stall:
            max_stall = delta
        if delta > 0.001:
            stalls += 1
    result.iterations = iterations
    result.seconds = time.perf_counter() - start
    result.max_stall_ms = max_stall * 1000.0
    result.stalls_over_1ms = stalls


def probe_gil(pipe: genai.ContinuousBatchingPipeline, cfg: genai.GenerationConfig) -> None:
    print("\n" + "=" * 72)
    print("EXPERIMENT 1 - does step() release the GIL?")
    print("=" * 72)

    # Baseline: monitor alone. The main thread is in time.sleep(), which
    # releases the GIL, so this is the monitor's uncontended rate.
    baseline = MonitorResult()
    stop = threading.Event()
    mon = threading.Thread(target=_run_monitor, args=(stop, baseline), daemon=True)
    mon.start()
    time.sleep(BASELINE_SECONDS)
    stop.set()
    mon.join()
    print(f"[baseline] monitor idle: {baseline.iters_per_sec:,.0f} iters/s, "
          f"max stall {baseline.max_stall_ms:.2f} ms")

    # Submit requests.
    handles: dict[int, genai.GenerationHandle] = {}
    for rid, prompt in enumerate(PROMPTS):
        handles[rid] = pipe.add_request(rid, prompt, cfg)
    token_counts: dict[int, int] = {}

    # Stepping phase: monitor runs while the MAIN thread drives step().
    stepping = MonitorResult()
    stop = threading.Event()
    mon = threading.Thread(target=_run_monitor, args=(stop, stepping), daemon=True)
    mon.start()

    step_times: list[float] = []
    phase_start = time.perf_counter()
    while pipe.has_non_finished_requests():
        t0 = time.perf_counter()
        pipe.step()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        for rid in drain_handles(handles, token_counts):
            handles.pop(rid, None)
    phase_seconds = time.perf_counter() - phase_start

    stop.set()
    mon.join()

    n_steps = len(step_times)
    avg_step = sum(step_times) / n_steps if n_steps else 0.0
    max_step = max(step_times) if step_times else 0.0
    print(f"[stepping] {n_steps} steps in {phase_seconds:.1f}s "
          f"(avg {avg_step:.1f} ms, max {max_step:.1f} ms)")
    print(f"[stepping] monitor while stepping: {stepping.iters_per_sec:,.0f} iters/s, "
          f"max stall {stepping.max_stall_ms:.2f} ms")
    print(f"[stepping] tokens generated: {sum(token_counts.values())}")

    ratio = (stepping.iters_per_sec / baseline.iters_per_sec
             if baseline.iters_per_sec > 0 else 0.0)
    print(f"\n[result] monitor kept {ratio * 100:.1f}% of its baseline rate "
          f"while step() ran.")
    print(f"[result] worst monitor stall while stepping: {stepping.max_stall_ms:.1f} ms "
          f"(slowest single step: {max_step:.1f} ms)")
    if ratio > 0.6:
        print("[verdict] step() RELEASES the GIL during the forward pass.")
        print("          -> run_in_executor / a worker thread genuinely frees the")
        print("             asyncio event loop. The recommended design holds.")
    elif ratio < 0.2:
        print("[verdict] step() HOLDS the GIL.")
        print("          -> a worker thread will NOT free the event loop; the CB")
        print("             engine loop would need a separate process.")
    else:
        print("[verdict] AMBIGUOUS - partial GIL release. Compare worst stall to")
        print("          slowest step above and re-run with a larger MAX_NEW_TOKENS.")


# --------------------------------------------------------------------------- #
# Experiment 2 - asyncio responsiveness with a single-worker executor         #
# --------------------------------------------------------------------------- #

async def _engine_loop(
    pipe: genai.ContinuousBatchingPipeline,
    executor: ThreadPoolExecutor,
    handles: dict[int, genai.GenerationHandle],
    token_counts: dict[int, int],
    done: asyncio.Event,
) -> None:
    loop = asyncio.get_running_loop()

    def step_and_drain() -> list[int]:
        pipe.step()
        return drain_handles(handles, token_counts)

    while pipe.has_non_finished_requests():
        finished = await loop.run_in_executor(executor, step_and_drain)
        for rid in finished:
            handles.pop(rid, None)
    done.set()


async def _heartbeat(done: asyncio.Event, jitter_ms: list[float]) -> None:
    target = HEARTBEAT_TARGET_MS / 1000.0
    while not done.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(target)
        jitter_ms.append((time.perf_counter() - t0 - target) * 1000.0)


async def _injector(
    pipe: genai.ContinuousBatchingPipeline,
    cfg: genai.GenerationConfig,
    executor: ThreadPoolExecutor,
    handles: dict[int, genai.GenerationHandle],
    start_rid: int,
) -> None:
    # Assumes generation outlasts the inject schedule (true for the default
    # MAX_NEW_TOKENS on an 8B model). Adds are routed through the same
    # single-worker executor so every pipeline call stays on one thread.
    loop = asyncio.get_running_loop()
    for offset, prompt in enumerate(INJECT_PROMPTS):
        await asyncio.sleep(1.0)
        rid = start_rid + offset
        handle = await loop.run_in_executor(executor, pipe.add_request, rid, prompt, cfg)
        handles[rid] = handle
        print(f"[inject] added request {rid} mid-run")


async def probe_asyncio(pipe: genai.ContinuousBatchingPipeline, cfg: genai.GenerationConfig) -> None:
    print("\n" + "=" * 72)
    print("EXPERIMENT 2 - asyncio responsiveness with single-worker executor")
    print("=" * 72)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cb-engine")
    loop = asyncio.get_running_loop()
    handles: dict[int, genai.GenerationHandle] = {}
    token_counts: dict[int, int] = {}

    # Disjoint id range from experiment 1 to avoid any duplicate-id surprises.
    for offset, prompt in enumerate(PROMPTS):
        rid = 1000 + offset
        handles[rid] = await loop.run_in_executor(executor, pipe.add_request, rid, prompt, cfg)

    done = asyncio.Event()
    jitter_ms: list[float] = []

    t0 = time.perf_counter()
    await asyncio.gather(
        _engine_loop(pipe, executor, handles, token_counts, done),
        _heartbeat(done, jitter_ms),
        _injector(pipe, cfg, executor, handles, start_rid=2000),
    )
    wall = time.perf_counter() - t0
    executor.shutdown(wait=True)

    jitter_ms.sort()
    n = len(jitter_ms)
    if n:
        p50 = jitter_ms[n // 2]
        p99 = jitter_ms[min(n - 1, int(n * 0.99))]
        worst = jitter_ms[-1]
    else:
        p50 = p99 = worst = 0.0

    print(f"[run] completed in {wall:.1f}s, "
          f"{sum(token_counts.values())} tokens over {len(token_counts)} requests")
    print(f"[heartbeat] {n} beats, target {HEARTBEAT_TARGET_MS:.0f} ms, jitter over "
          f"target: p50 {p50:.1f} ms, p99 {p99:.1f} ms, worst {worst:.1f} ms")

    if worst < 25.0:
        print("[verdict] event loop stayed RESPONSIVE while generation ran.")
        print("          a single-worker ThreadPoolExecutor is a sound base for")
        print("          the CB engine loop.")
    else:
        print("[verdict] event loop saw STALLS - worst heartbeat jitter is high.")
        print("          if experiment 1 said the GIL is released, suspect the")
        print("          executor handoff or per-step Python work; if the GIL is")
        print("          held, that is the cause and a thread will not fix it.")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    pipe = build_pipeline()
    cfg = build_generation_config()

    probe_gil(pipe, cfg)
    asyncio.run(probe_asyncio(pipe, cfg))

    print("\n" + "=" * 72)
    print("Done. Experiment 1's verdict is the load-bearing one: it decides")
    print("whether the CB engine loop can live in a thread (recommended) or")
    print("must run in a separate process.")
    print("=" * 72)


if __name__ == "__main__":
    main()