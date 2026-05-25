"""
Out-of-process CB engine - proof and blueprint.

cb_server_block_test.py showed an in-process CB engine freezes the whole
server: a do-nothing /ping went from ~1 ms to ~389 ms while a batch ran,
because step() holds the GIL.

This script runs the SAME load test, but the CB engine lives in its own
process, spawned (not forked). The FastAPI event loop never calls step() -
it only puts small messages on a multiprocessing.Queue and reads results
back through a thread that blocks on IPC (which releases the GIL while
waiting). /ping should stay flat at ~1 ms even while a batch is in flight.

It is also the blueprint for the OpenArc integration:
  - engine_process()        == the CB engine loop (runs in the child)
  - the FastAPI server side == what queue_worker_cb becomes (the bridge)

Process layout:
    this script (client) --spawns--> FastAPI server --spawns--> engine
The client/server split only isolates the client's measurements.
The server/engine split is the actual fix.

Run on the GPU box:
    pip install fastapi uvicorn httpx
    python cb_server_oop_test.py
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI


# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #

MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Anubis-Mini-8B-v1-int4_asym-ov/"
DEVICE = "GPU.0"
HOST = "127.0.0.1"
PORT = 8138

SCHEDULER = {
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 16,
    "cache_size": 8,                  # tune to your VRAM
    "dynamic_split_fuse": True,
    "enable_prefix_caching": True,
}

MAX_NEW_TOKENS = 256
BATCH_SIZE = 6
BASELINE_PINGS = 60
SLOW_PING_MS = 50.0

PROMPTS = [
    "Explain how continuous batching differs from static batching for LLM serving.",
    "Write a short paragraph about the history of the printing press.",
    "Describe, step by step, how a CPU executes a single instruction.",
    "Summarize the causes of the 2008 financial crisis in plain language.",
    "List five considerations when choosing a database index, with reasons.",
    "Explain what a KV cache is and why it matters for transformer inference.",
]


# =========================================================================== #
# ENGINE PROCESS - runs in its own interpreter, spawned by the server.         #
# Everything below in this section executes in the CHILD.                      #
# =========================================================================== #

def _build_pipeline(model_path, device, scheduler):
    import openvino_genai as genai

    sc = genai.SchedulerConfig()
    for key, value in scheduler.items():
        setattr(sc, key, value)
    return genai.ContinuousBatchingPipeline(model_path, device=device, scheduler_config=sc)


def _handle_request(msg, pipe, handles, genai) -> bool:
    """Apply one inbound message. Returns False on shutdown."""
    kind = msg["type"]
    if kind == "shutdown":
        return False
    if kind == "cancel":
        handle = handles.get(msg["rid"])
        if handle is not None:
            handle.cancel()
    elif kind == "submit":
        # In the real server, send the whole gen-config as a dict and
        # rebuild it here. The test only varies max_new_tokens.
        cfg = genai.GenerationConfig()
        cfg.max_new_tokens = msg["max_new_tokens"]
        cfg.do_sample = False
        handles[msg["rid"]] = pipe.add_request(msg["rid"], msg["prompt"], cfg)
    return True


def engine_process(model_path, device, scheduler, req_q, res_q) -> None:
    """The CB engine loop. Pure synchronous code, its own GIL, no asyncio."""
    import openvino_genai as genai

    terminal = {
        genai.GenerationStatus.FINISHED,
        genai.GenerationStatus.IGNORED,
        genai.GenerationStatus.CANCEL,
        genai.GenerationStatus.STOP,
    }

    try:
        print(f"[engine] loading {model_path} on {device} ...", flush=True)
        t0 = time.perf_counter()
        pipe = _build_pipeline(model_path, device, scheduler)
        print(f"[engine] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - report any load failure to the parent
        res_q.put({"type": "load_error", "error": repr(exc)})
        return

    res_q.put({"type": "ready"})

    handles: dict[int, "genai.GenerationHandle"] = {}
    running = True
    try:
        while running:
            # Idle: block on the request queue. Zero CPU spin, instant wake.
            if not handles:
                running = _handle_request(req_q.get(), pipe, handles, genai)
                if not running:
                    break
            # Drain any other queued messages without blocking.
            while True:
                try:
                    msg = req_q.get_nowait()
                except queue.Empty:
                    break
                if not _handle_request(msg, pipe, handles, genai):
                    running = False
                    break
            if not running or not handles:
                continue

            # One inference step across every in-flight request.
            pipe.step()

            # Hand finished/streaming tokens back to the parent.
            for rid in list(handles.keys()):
                handle = handles[rid]
                if handle.can_read():
                    token_ids: list[int] = []
                    for output in handle.read().values():
                        token_ids.extend(output.generated_ids)
                    if token_ids:
                        res_q.put({"type": "tokens", "rid": rid, "token_ids": token_ids})
                status = handle.get_status()
                if status in terminal:
                    res_q.put({"type": "done", "rid": rid, "status": status.name})
                    del handles[rid]
    except Exception as exc:  # noqa: BLE001
        res_q.put({"type": "engine_error", "error": repr(exc)})
    finally:
        for handle in handles.values():
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001
                pass
        print("[engine] stopped", flush=True)


# =========================================================================== #
# FASTAPI SERVER - the parent. This is the "bridge" queue_worker_cb becomes.    #
# =========================================================================== #

_req_q: mp.Queue | None = None
_res_q: mp.Queue | None = None
_proc: mp.Process | None = None
_reader: threading.Thread | None = None

_futures: dict[int, asyncio.Future] = {}
_token_counts: dict[int, int] = {}
_rid_counter = itertools.count()


def _dispatch_result(msg: dict) -> None:
    """Runs on the event loop (via call_soon_threadsafe). Routes one result."""
    kind = msg.get("type")
    rid = msg.get("rid")
    if kind == "tokens":
        _token_counts[rid] = _token_counts.get(rid, 0) + len(msg["token_ids"])
    elif kind == "done":
        fut = _futures.pop(rid, None)
        if fut is not None and not fut.done():
            if msg["status"] == "IGNORED":
                fut.set_exception(RuntimeError("request IGNORED - KV cache OOM"))
            else:
                fut.set_result(_token_counts.pop(rid, 0))
    elif kind == "engine_error":
        print(f"[server] ENGINE ERROR: {msg['error']}", flush=True)


def _result_reader(loop: asyncio.AbstractEventLoop, res_q: mp.Queue) -> None:
    """Dedicated thread: blocks on res_q.get() (releases the GIL while waiting,
    so it does NOT starve the event loop) and forwards each result."""
    while True:
        msg = res_q.get()
        if msg.get("type") == "_stop":
            break
        loop.call_soon_threadsafe(_dispatch_result, msg)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _req_q, _res_q, _proc, _reader
    ctx = mp.get_context("spawn")  # spawn, NOT fork - GPU state must not be forked
    _req_q = ctx.Queue()
    _res_q = ctx.Queue()
    _proc = ctx.Process(
        target=engine_process,
        args=(MODEL_PATH, DEVICE, SCHEDULER, _req_q, _res_q),
        daemon=True,
    )
    _proc.start()

    loop = asyncio.get_running_loop()
    # First message from the child is the load result; wait for it off-loop.
    first = await loop.run_in_executor(None, _res_q.get)
    if first["type"] == "load_error":
        raise RuntimeError(f"engine failed to load: {first['error']}")

    _reader = threading.Thread(target=_result_reader, args=(loop, _res_q), daemon=True)
    _reader.start()
    print(f"[server] listening on http://{HOST}:{PORT}", flush=True)
    yield

    _req_q.put({"type": "shutdown"})
    _proc.join(timeout=10)
    if _proc.is_alive():
        _proc.terminate()
    _res_q.put({"type": "_stop"})
    if _reader is not None:
        _reader.join(timeout=2)


app = FastAPI(lifespan=_lifespan)


@app.get("/ping")
async def ping():
    """Does nothing. If this is slow, the event loop was blocked."""
    return {"t": time.perf_counter()}


@app.post("/generate")
async def generate(idx: int = 0):
    loop = asyncio.get_running_loop()
    rid = next(_rid_counter)
    fut: asyncio.Future = loop.create_future()
    _futures[rid] = fut
    # queue.put is fast and does not block the event loop.
    _req_q.put({
        "type": "submit",
        "rid": rid,
        "prompt": PROMPTS[idx % len(PROMPTS)],
        "max_new_tokens": MAX_NEW_TOKENS,
    })
    tokens = await fut
    return {"rid": rid, "tokens": tokens}


def run_server() -> None:
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    uvicorn.Server(config).run()


# =========================================================================== #
# CLIENT - load test (identical to cb_server_block_test.py for comparison)      #
# =========================================================================== #

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
        except Exception:  # noqa: BLE001
            break
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


async def load_test(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        print("[test] measuring baseline /ping latency (server idle) ...")
        baseline = await _measure_pings(client, BASELINE_PINGS)

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
    print("RESULT - engine OUT OF PROCESS")
    print("=" * 72)
    print(f"batch: {BATCH_SIZE} requests, {total_tokens} tokens generated")
    print(f"/ping  server IDLE        : p50={b_p50:6.1f}ms  p99={b_p99:7.1f}ms  "
          f"max={b_max:7.1f}ms   ({b_n} pings)")
    print(f"/ping  BATCH IN FLIGHT    : p50={d_p50:6.1f}ms  p99={d_p99:7.1f}ms  "
          f"max={d_max:7.1f}ms   ({d_n} pings)")
    print(f"                          : {slow} of {d_n} pings delayed past {SLOW_PING_MS:.0f}ms")

    stayed_responsive = d_max < 25.0 or d_max < 3.0 * max(b_max, 1.0)
    print()
    if stayed_responsive:
        print("[verdict] FIXED. /ping stayed responsive while a batch ran - the")
        print("          event loop is free because step() runs in another")
        print("          process. Compare this max to the in-process test.")
    else:
        print("[verdict] /ping is still slow. Check the engine really ran in the")
        print("          child process and that nothing heavy is on the loop.")
    print("=" * 72)


def run_test() -> None:
    base_url = f"http://{HOST}:{PORT}"
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"])
    try:
        print("[test] waiting for server (engine spawn + model load, ~10-30s) ...")
        deadline = time.time() + 240.0
        with httpx.Client() as probe:
            while True:
                try:
                    if probe.get(base_url + "/ping", timeout=2.0).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
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