"""
Process isolation A/B test - in-process vs out-of-process CB engine.

This is cb_server_block_test.py and cb_server_zmq_test.py merged into one
harness. It runs the SAME FastAPI load test twice:

  mode "inprocess"  - CB engine runs on the server's event loop, step()
                      driven through a ThreadPoolExecutor(max_workers=1).
                      This is the design that blocks.
  mode "zmq"        - CB engine runs in a separate spawned process; the
                      server talks to it over ZeroMQ. step() never touches
                      the event loop.

Both modes expose identical /ping and /generate endpoints and run the
identical client load test, so the only variable is WHERE step() executes.
A do-nothing /ping is the canary: if it slows while a batch is in flight,
the event loop was blocked.

Process layout per mode:
    driver --spawns--> FastAPI server  (--spawns--> ZMQ engine, zmq mode only)
The two modes run sequentially, so only one holds GPU.0 at a time.

Targets Python 3.12. Run on the GPU box:
    pip install pyzmq msgspec fastapi uvicorn httpx
    python cb_isolation_ab_test.py
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import msgspec
import openvino_genai as genai
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI


# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #

MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Anubis-Mini-8B-v1-int4_asym-ov/"
DEVICE = "GPU.0"
HOST = "127.0.0.1"
PORT_INPROC = 8140
PORT_ZMQ = 8141

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
ZMQ_LINGER_MS = 1000
GPU_RELEASE_GAP_S = 5.0               # wait between modes so VRAM frees

PROMPTS = [
    "Explain how continuous batching differs from static batching for LLM serving.",
    "Write a short paragraph about the history of the printing press.",
    "Describe, step by step, how a CPU executes a single instruction.",
    "Summarize the causes of the 2008 financial crisis in plain language.",
    "List five considerations when choosing a database index, with reasons.",
    "Explain what a KV cache is and why it matters for transformer inference.",
]

TERMINAL = {
    genai.GenerationStatus.FINISHED,
    genai.GenerationStatus.IGNORED,
    genai.GenerationStatus.CANCEL,
    genai.GenerationStatus.STOP,
}


# --------------------------------------------------------------------------- #
# SHARED - pipeline build + per-request future bookkeeping                      #
# --------------------------------------------------------------------------- #

def _build_pipeline(model_path, device, scheduler):
    sc = genai.SchedulerConfig()
    for key, value in scheduler.items():
        setattr(sc, key, value)
    return genai.ContinuousBatchingPipeline(model_path, device=device, scheduler_config=sc)


_futures: dict[int, asyncio.Future] = {}
_token_counts: dict[int, int] = {}
_rid_counter = itertools.count()


def _resolve(rid: int, status_name: str) -> None:
    """Settle one request's future. Used by both backends."""
    fut = _futures.pop(rid, None)
    if fut is not None and not fut.done():
        if status_name == "IGNORED":
            fut.set_exception(RuntimeError("request IGNORED - KV cache OOM"))
        else:
            fut.set_result(_token_counts.pop(rid, 0))


# =========================================================================== #
# ZMQ ENGINE PROCESS - used by the "zmq" mode. Runs in its own interpreter.     #
# =========================================================================== #

def _handle_request(msg, pipe, handles) -> bool:
    """Apply one inbound message in the engine process. False on shutdown."""
    kind = msg["type"]
    if kind == "shutdown":
        return False
    if kind == "cancel":
        handle = handles.get(msg["rid"])
        if handle is not None:
            handle.cancel()
    elif kind == "submit":
        cfg = genai.GenerationConfig()
        cfg.max_new_tokens = msg["max_new_tokens"]
        cfg.do_sample = False
        handles[msg["rid"]] = pipe.add_request(msg["rid"], msg["prompt"], cfg)
    return True


def engine_process(model_path, device, scheduler, req_endpoint, res_endpoint) -> None:
    """The CB engine loop for zmq mode. Pure synchronous code, its own GIL."""
    ctx = zmq.Context()
    req_sock = ctx.socket(zmq.PULL)
    req_sock.connect(req_endpoint)
    res_sock = ctx.socket(zmq.PUSH)
    res_sock.setsockopt(zmq.LINGER, ZMQ_LINGER_MS)
    res_sock.connect(res_endpoint)

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    def send(msg: dict) -> None:
        res_sock.send(encoder.encode(msg))

    try:
        print(f"[engine] loading {model_path} on {device} ...", flush=True)
        t0 = time.perf_counter()
        pipe = _build_pipeline(model_path, device, scheduler)
        print(f"[engine] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        send({"type": "load_error", "error": repr(exc)})
        res_sock.close()
        req_sock.close()
        ctx.term()
        return

    send({"type": "ready"})

    handles: dict[int, "genai.GenerationHandle"] = {}
    running = True
    try:
        while running:
            if not handles:
                running = _handle_request(decoder.decode(req_sock.recv()), pipe, handles)
                if not running:
                    break
            while True:
                try:
                    raw = req_sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if not _handle_request(decoder.decode(raw), pipe, handles):
                    running = False
                    break
            if not running or not handles:
                continue

            pipe.step()

            for rid in list(handles.keys()):
                handle = handles[rid]
                if handle.can_read():
                    token_ids: list[int] = []
                    for output in handle.read().values():
                        token_ids.extend(output.generated_ids)
                    if token_ids:
                        send({"type": "tokens", "rid": rid, "token_ids": token_ids})
                status = handle.get_status()
                if status in TERMINAL:
                    send({"type": "done", "rid": rid, "status": status.name})
                    del handles[rid]
    except Exception as exc:  # noqa: BLE001
        send({"type": "engine_error", "error": repr(exc)})
    finally:
        for handle in handles.values():
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001
                pass
        res_sock.close()
        req_sock.close()
        ctx.term()
        print("[engine] stopped", flush=True)


# =========================================================================== #
# BACKEND A - in-process engine (the design under test as "blocking")          #
# =========================================================================== #

class InProcessBackend:
    """CB engine on the server's event loop. step() via a 1-worker executor."""

    def __init__(self) -> None:
        self.pipe = None
        self.cfg = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cb-engine")
        self.handles: dict[int, "genai.GenerationHandle"] = {}
        self.wake = asyncio.Event()
        self.loop_task: asyncio.Task | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.pipe = await loop.run_in_executor(
            self.executor, _build_pipeline, MODEL_PATH, DEVICE, SCHEDULER
        )
        self.cfg = genai.GenerationConfig()
        self.cfg.max_new_tokens = MAX_NEW_TOKENS
        self.cfg.do_sample = False
        self.loop_task = asyncio.create_task(self._engine_loop())

    def _step_and_drain(self) -> list[tuple[int, str]]:
        self.pipe.step()
        finished: list[tuple[int, str]] = []
        for rid, handle in self.handles.items():
            if handle.can_read():
                for output in handle.read().values():
                    _token_counts[rid] = _token_counts.get(rid, 0) + len(output.generated_ids)
            status = handle.get_status()
            if status in TERMINAL:
                finished.append((rid, status.name))
        return finished

    async def _engine_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                if not self.handles:
                    self.wake.clear()
                    await self.wake.wait()
                finished = await loop.run_in_executor(self.executor, self._step_and_drain)
                for rid, status_name in finished:
                    self.handles.pop(rid, None)
                    _resolve(rid, status_name)
        except asyncio.CancelledError:
            pass

    async def submit(self, rid: int, prompt: str, max_new_tokens: int) -> None:
        loop = asyncio.get_running_loop()
        handle = await loop.run_in_executor(
            self.executor, self.pipe.add_request, rid, prompt, self.cfg
        )
        self.handles[rid] = handle
        self.wake.set()

    async def stop(self) -> None:
        if self.loop_task is not None:
            self.loop_task.cancel()
            await asyncio.gather(self.loop_task, return_exceptions=True)
        self.executor.shutdown(wait=False)


# =========================================================================== #
# BACKEND B - out-of-process engine over ZeroMQ                                 #
# =========================================================================== #

def _dispatch_result(msg: dict) -> None:
    kind = msg.get("type")
    rid = msg.get("rid")
    if kind == "tokens":
        _token_counts[rid] = _token_counts.get(rid, 0) + len(msg["token_ids"])
    elif kind == "done":
        _resolve(rid, msg["status"])
    elif kind == "engine_error":
        print(f"[server] ENGINE ERROR: {msg['error']}", flush=True)


class ZmqBackend:
    """CB engine in a spawned child process, reached over ZeroMQ."""

    def __init__(self) -> None:
        self.ctx: zmq.asyncio.Context | None = None
        self.req_sock: zmq.asyncio.Socket | None = None
        self.res_sock: zmq.asyncio.Socket | None = None
        self.outbox: asyncio.Queue | None = None
        self.proc: mp.Process | None = None
        self.tasks: list[asyncio.Task] = []
        self.endpoints: tuple[str, str] = ("", "")
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()

    async def start(self) -> None:
        req_ep = f"ipc:///tmp/openarc_cb_req_{os.getpid()}.ipc"
        res_ep = f"ipc:///tmp/openarc_cb_res_{os.getpid()}.ipc"
        self.endpoints = (req_ep, res_ep)

        self.ctx = zmq.asyncio.Context()
        self.req_sock = self.ctx.socket(zmq.PUSH)
        self.req_sock.setsockopt(zmq.LINGER, ZMQ_LINGER_MS)
        self.req_sock.bind(req_ep)
        self.res_sock = self.ctx.socket(zmq.PULL)
        self.res_sock.setsockopt(zmq.LINGER, ZMQ_LINGER_MS)
        self.res_sock.bind(res_ep)
        self.outbox = asyncio.Queue()

        mpctx = mp.get_context("spawn")  # spawn, NOT fork - GPU state
        self.proc = mpctx.Process(
            target=engine_process,
            args=(MODEL_PATH, DEVICE, SCHEDULER, req_ep, res_ep),
            daemon=True,
        )
        self.proc.start()

        first = self.decoder.decode(await self.res_sock.recv())
        if first["type"] == "load_error":
            raise RuntimeError(f"engine failed to load: {first['error']}")

        self.tasks = [
            asyncio.create_task(self._request_sender()),
            asyncio.create_task(self._result_handler()),
        ]

    async def _request_sender(self) -> None:
        while True:
            msg = await self.outbox.get()
            try:
                await self.req_sock.send(self.encoder.encode(msg))
            finally:
                self.outbox.task_done()

    async def _result_handler(self) -> None:
        while True:
            raw = await self.res_sock.recv()
            _dispatch_result(self.decoder.decode(raw))

    async def submit(self, rid: int, prompt: str, max_new_tokens: int) -> None:
        await self.outbox.put({
            "type": "submit", "rid": rid, "prompt": prompt, "max_new_tokens": max_new_tokens,
        })

    async def stop(self) -> None:
        await self.outbox.put({"type": "shutdown"})
        await self.outbox.join()
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.req_sock.close()
        self.res_sock.close()
        self.ctx.term()
        for endpoint in self.endpoints:
            try:
                os.unlink(endpoint.removeprefix("ipc://"))
            except OSError:
                pass


# =========================================================================== #
# FASTAPI SERVER - mode-agnostic endpoints over whichever backend is selected   #
# =========================================================================== #

MODE = "inprocess"
_backend: InProcessBackend | ZmqBackend | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _backend
    _backend = InProcessBackend() if MODE == "inprocess" else ZmqBackend()
    await _backend.start()
    print(f"[server] mode={MODE} ready", flush=True)
    yield
    await _backend.stop()


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
    await _backend.submit(rid, PROMPTS[idx % len(PROMPTS)], MAX_NEW_TOKENS)
    tokens = await fut
    return {"rid": rid, "tokens": tokens}


def run_server(mode: str, port: int) -> None:
    global MODE
    MODE = mode
    config = uvicorn.Config(app, host=HOST, port=port, log_level="warning")
    uvicorn.Server(config).run()


# =========================================================================== #
# CLIENT - load test (returns stats so the driver can compare modes)            #
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


async def load_test(base_url: str) -> dict:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        print("[test]   baseline /ping (server idle) ...")
        baseline = await _measure_pings(client, BASELINE_PINGS)

        print(f"[test]   firing {BATCH_SIZE} /generate, pinging during the batch ...")
        gen_task = asyncio.ensure_future(
            asyncio.gather(*[client.post("/generate", params={"idx": i}) for i in range(BATCH_SIZE)])
        )
        during = await _measure_pings_until(client, gen_task)
        gen_results = await gen_task

    b_p50, b_p99, b_max, _ = _stats(baseline)
    d_p50, d_p99, d_max, d_n = _stats(during)
    return {
        "idle_p50": b_p50, "idle_p99": b_p99, "idle_max": b_max,
        "batch_p50": d_p50, "batch_p99": d_p99, "batch_max": d_max, "batch_n": d_n,
        "slow": sum(1 for x in during if x > SLOW_PING_MS),
        "tokens": sum(r.json().get("tokens", 0) for r in gen_results),
    }


# --------------------------------------------------------------------------- #
# DRIVER - run both modes, print the side-by-side comparison                    #
# --------------------------------------------------------------------------- #

def _run_one_mode(mode: str, port: int) -> dict:
    base_url = f"http://{HOST}:{port}"
    print(f"\n[driver] starting server in mode={mode} ...")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve", mode, str(port)])
    try:
        deadline = time.time() + 240.0
        with httpx.Client() as probe:
            while True:
                try:
                    if probe.get(base_url + "/ping", timeout=2.0).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    pass
                if time.time() > deadline:
                    raise RuntimeError(f"server (mode={mode}) did not become ready")
                time.sleep(1.0)
        print(f"[driver] mode={mode} server ready, running load test ...")
        return asyncio.run(load_test(base_url))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _fmt(value: float) -> str:
    return f"{value:8.1f}ms"


def _print_comparison(inproc: dict, zmqd: dict) -> None:
    rows = [
        ("/ping idle      max", "idle_max"),
        ("/ping in-batch  p50", "batch_p50"),
        ("/ping in-batch  p99", "batch_p99"),
        ("/ping in-batch  max", "batch_max"),
    ]
    print("\n" + "=" * 72)
    print("COMPARISON - process isolation during step()")
    print("=" * 72)
    print(f"{'':22} {'IN-PROCESS':>14}   {'OUT-OF-PROCESS (ZMQ)':>22}")
    for label, key in rows:
        print(f"{label:22} {_fmt(inproc[key]):>14}   {_fmt(zmqd[key]):>22}")
    print(f"{'pings delayed >'+str(int(SLOW_PING_MS))+'ms':22} "
          f"{str(inproc['slow'])+'/'+str(inproc['batch_n']):>14}   "
          f"{str(zmqd['slow'])+'/'+str(zmqd['batch_n']):>22}")
    print("-" * 72)

    inproc_blocked = inproc["batch_max"] > 25.0 and inproc["batch_max"] > 5.0 * max(inproc["idle_max"], 1.0)
    zmq_clean = zmqd["batch_max"] < 25.0 or zmqd["batch_max"] < 3.0 * max(zmqd["idle_max"], 1.0)
    if inproc_blocked and zmq_clean:
        print("[verdict] CONFIRMED. In-process, a do-nothing /ping spiked to "
              f"{inproc['batch_max']:.0f}ms while a batch ran -")
        print("          the event loop was blocked by step(). Out-of-process over")
        print(f"          ZMQ, /ping stayed at {zmqd['batch_max']:.0f}ms max. Process")
        print("          isolation keeps FastAPI free during step(): the engine's")
        print("          GIL is a different GIL.")
    else:
        print("[verdict] inconclusive - inspect the numbers above. Expected the")
        print("          in-process max to be large and the ZMQ max to stay small.")
    print("=" * 72)


def run_test() -> None:
    inproc = _run_one_mode("inprocess", PORT_INPROC)
    print(f"\n[driver] waiting {GPU_RELEASE_GAP_S:.0f}s for GPU memory to free ...")
    time.sleep(GPU_RELEASE_GAP_S)
    zmqd = _run_one_mode("zmq", PORT_ZMQ)
    _print_comparison(inproc, zmqd)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if "--serve" in sys.argv:
        i = sys.argv.index("--serve")
        run_server(sys.argv[i + 1], int(sys.argv[i + 2]))
    else:
        run_test()