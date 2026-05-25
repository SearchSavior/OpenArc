"""
Does an in-process CB engine block the rest of the server?

This stands up a REAL FastAPI + uvicorn server whose CB engine loop runs
the *recommended* in-process design - step() driven through a
ThreadPoolExecutor(max_workers=1) on the server's event loop. Then a
separate client process measures the latency of a do-nothing /ping
endpoint, first while the server is idle, then while a generation batch
is in flight.

If /ping - which does nothing but return a timestamp - slows from ~1 ms
to hundreds of ms while a batch runs, that is every other request and
every SSE stream on the server stalling for the duration of each step().

Run on the machine with the GPU:

    pip install fastapi uvicorn httpx        # if not already present
    python cb_server_block_test.py

The script spawns its own server subprocess; you do not start one
manually. NOTE: the subprocess split here is only to isolate the client's
measurements from the server - the *engine* still runs in-process to the
server. That in-process engine is exactly the design under test.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import openvino_genai as genai
import uvicorn
from fastapi import FastAPI


# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #

MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Anubis-Mini-8B-v1-int4_asym-ov/"
DEVICE = "GPU.0"
HOST = "127.0.0.1"
PORT = 8137

SCHED_MAX_NUM_BATCHED_TOKENS = 2048
SCHED_MAX_NUM_SEQS = 16
SCHED_CACHE_SIZE_GB = 8           # tune to your VRAM
SCHED_DYNAMIC_SPLIT_FUSE = True
SCHED_ENABLE_PREFIX_CACHING = True

MAX_NEW_TOKENS = 256              # long enough that many pings land during the batch
BATCH_SIZE = 6                    # /generate requests fired at once
BASELINE_PINGS = 60               # pings measured while the server is idle
SLOW_PING_MS = 50.0               # a /ping over this is "delayed"

PROMPTS = [
    "Explain how continuous batching differs from static batching for LLM serving.",
    "Write a short paragraph about the history of the printing press.",
    "Describe, step by step, how a CPU executes a single instruction.",
    "Summarize the causes of the 2008 financial crisis in plain language.",
    "List five considerations when choosing a database index, with reasons.",
    "Explain what a KV cache is and why it matters for transformer inference.",
]

TERMINAL_STATUSES = {
    genai.GenerationStatus.FINISHED,
    genai.GenerationStatus.IGNORED,
    genai.GenerationStatus.CANCEL,
    genai.GenerationStatus.STOP,
}


# --------------------------------------------------------------------------- #
# SERVER SIDE - FastAPI app with an in-process CB engine                       #
# --------------------------------------------------------------------------- #

_pipe: genai.ContinuousBatchingPipeline | None = None
_cfg: genai.GenerationConfig | None = None
_engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cb-engine")

_handles: dict[int, genai.GenerationHandle] = {}
_futures: dict[int, asyncio.Future] = {}
_token_counts: dict[int, int] = {}
_wake = asyncio.Event()
_rid_counter = itertools.count()


def _build_pipeline() -> genai.ContinuousBatchingPipeline:
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = SCHED_MAX_NUM_BATCHED_TOKENS
    scheduler.max_num_seqs = SCHED_MAX_NUM_SEQS
    scheduler.cache_size = SCHED_CACHE_SIZE_GB
    scheduler.dynamic_split_fuse = SCHED_DYNAMIC_SPLIT_FUSE
    scheduler.enable_prefix_caching = SCHED_ENABLE_PREFIX_CACHING

    print(f"[server] loading {MODEL_PATH} on {DEVICE} ...", flush=True)
    t0 = time.perf_counter()
    pipe = genai.ContinuousBatchingPipeline(
        MODEL_PATH, device=DEVICE, scheduler_config=scheduler
    )
    print(f"[server] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def _build_generation_config() -> genai.GenerationConfig:
    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    cfg.do_sample = False
    return cfg


def _step_and_drain() -> list[int]:
    """Runs on the single-worker executor thread. step() then drain."""
    _pipe.step()
    finished: list[int] = []
    for rid, handle in _handles.items():
        if handle.can_read():
            for output in handle.read().values():
                _token_counts[rid] = _token_counts.get(rid, 0) + len(output.generated_ids)
        if handle.get_status() in TERMINAL_STATUSES:
            finished.append(rid)
    return finished


async def _engine_loop() -> None:
    """The recommended in-process design: step() via run_in_executor on a
    single-worker pool, so the call is off the event loop thread."""
    loop = asyncio.get_running_loop()
    try:
        while True:
            if not _handles:
                _wake.clear()
                await _wake.wait()
            finished = await loop.run_in_executor(_engine_executor, _step_and_drain)
            for rid in finished:
                _handles.pop(rid, None)
                fut = _futures.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(_token_counts.get(rid, 0))
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pipe, _cfg
    _pipe = _build_pipeline()
    _cfg = _build_generation_config()
    engine_task = asyncio.create_task(_engine_loop())
    print(f"[server] listening on http://{HOST}:{PORT}", flush=True)
    yield
    engine_task.cancel()
    _engine_executor.shutdown(wait=False)


app = FastAPI(lifespan=_lifespan)


@app.get("/ping")
async def ping():
    """Does nothing. If this is slow, the event loop was blocked."""
    return {"t": time.perf_counter()}


@app.post("/generate")
async def generate(idx: int = 0):
    """Submit one request to the CB engine and wait for it to finish."""
    loop = asyncio.get_running_loop()
    rid = next(_rid_counter)
    fut: asyncio.Future = loop.create_future()
    _futures[rid] = fut
    prompt = PROMPTS[idx % len(PROMPTS)]
    handle = await loop.run_in_executor(_engine_executor, _pipe.add_request, rid, prompt, _cfg)
    _handles[rid] = handle
    _wake.set()
    tokens = await fut
    return {"rid": rid, "tokens": tokens}


def run_server() -> None:
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    uvicorn.Server(config).run()


# --------------------------------------------------------------------------- #
# CLIENT SIDE - load test                                                      #
# --------------------------------------------------------------------------- #

def _stats(latencies_ms: list[float]) -> tuple[float, float, float, int]:
    if not latencies_ms:
        return 0.0, 0.0, 0.0, 0
    s = sorted(latencies_ms)
    n = len(s)
    return s[n // 2], s[min(n - 1, int(n * 0.99))], s[-1], n


async def _measure_pings(client: httpx.AsyncClient, n: int) -> list[float]:
    out: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await client.get("/ping")
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


async def _measure_pings_until(client: httpx.AsyncClient, until: asyncio.Future) -> list[float]:
    out: list[float] = []
    while not until.done():
        t0 = time.perf_counter()
        try:
            await client.get("/ping")
        except Exception:
            break
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


async def load_test(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        # 1. Baseline: /ping latency with the server idle.
        print("[test] measuring baseline /ping latency (server idle) ...")
        baseline = await _measure_pings(client, BASELINE_PINGS)

        # 2. Fire a batch of /generate, ping continuously while it runs.
        print(f"[test] firing {BATCH_SIZE} /generate requests, pinging during the batch ...")
        gen_task = asyncio.ensure_future(
            asyncio.gather(*[client.post("/generate", params={"idx": i}) for i in range(BATCH_SIZE)])
        )
        during = await _measure_pings_until(client, gen_task)
        gen_results = await gen_task

    b_p50, b_p99, b_max, b_n = _stats(baseline)
    d_p50, d_p99, d_max, d_n = _stats(during)
    slow = sum(1 for x in during if x > SLOW_PING_MS)
    total_tokens = sum(r.json()["tokens"] for r in gen_results)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"batch: {BATCH_SIZE} requests, {total_tokens} tokens generated")
    print(f"/ping  server IDLE        : p50={b_p50:6.1f}ms  p99={b_p99:7.1f}ms  "
          f"max={b_max:7.1f}ms   ({b_n} pings)")
    print(f"/ping  BATCH IN FLIGHT    : p50={d_p50:6.1f}ms  p99={d_p99:7.1f}ms  "
          f"max={d_max:7.1f}ms   ({d_n} pings)")
    print(f"                          : {slow} of {d_n} pings delayed past {SLOW_PING_MS:.0f}ms")

    blocked = d_max > 25.0 and (b_max == 0.0 or d_max > 10.0 * b_max)
    print()
    if blocked:
        print("[verdict] CONFIRMED. A do-nothing /ping endpoint went from "
              f"~{b_p50:.1f}ms idle to {d_max:.0f}ms while a batch ran.")
        print("          The in-process engine blocks the event loop: every other")
        print("          request and SSE stream stalls for the duration of each")
        print("          step(). The CB engine must run out-of-process.")
    else:
        print("[verdict] NOT reproduced - /ping stayed fast during the batch.")
        print("          Re-check that the engine actually ran in-process here.")
    print("=" * 72)


def run_test() -> None:
    base_url = f"http://{HOST}:{PORT}"
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"])
    try:
        print("[test] waiting for server (model load can take ~10-30s) ...")
        deadline = time.time() + 240.0
        with httpx.Client() as probe:
            while True:
                try:
                    if probe.get(base_url + "/ping", timeout=2.0).status_code == 200:
                        break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("server did not become ready in time")
                time.sleep(1.0)
        print("[test] server ready.\n")
        asyncio.run(load_test(base_url))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if "--serve" in sys.argv:
        run_server()
    else:
        run_test()