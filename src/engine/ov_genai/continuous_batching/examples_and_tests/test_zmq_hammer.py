"""
Concurrent-load hammer test - in-process vs out-of-process CB engine.

The earlier A/B test pinged sequentially, one request at a time. That
undersamples blocking: a ping issued in the gap between two step() calls
completes fast and never witnesses the freeze. This version fixes that -
it holds CONCURRENT_PINGERS (default 48) ping clients hammering /ping
continuously, so whenever the event loop freezes there are always dozens
of requests caught mid-flight. The slow ones cannot hide.

Each mode runs three phases against this sustained concurrent load:
  baseline   - hammer only, no generation (the concurrent-load floor)
  batch      - BATCH_SIZE /generate requests fired, hammer continues
  cooldown   - hammer only, after the batch finishes

Every /ping is recorded as (time_since_start, latency_ms). After both
modes run, the script writes:
  cb_hammer_results.png      - latency timeline (per mode) + CDF (overlaid)
  cb_hammer_inprocess.csv    - raw samples, in-process mode
  cb_hammer_zmq.csv          - raw samples, zmq mode

Targets Python 3.12. Run on the GPU box:
    pip install pyzmq msgspec fastapi uvicorn httpx matplotlib
    python cb_hammer_test.py
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
PORT_INPROC = 8142
PORT_ZMQ = 8143

SCHEDULER = {
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 16,
    "cache_size": 8,                  # tune to your VRAM
    "dynamic_split_fuse": True,
    "enable_prefix_caching": True,
}

MAX_NEW_TOKENS = 256
BATCH_SIZE = 6
CONCURRENT_PINGERS = 48               # sustained concurrent /ping clients
BASELINE_SECONDS = 5.0                # hammer-only phase before the batch
COOLDOWN_SECONDS = 3.0                # hammer-only phase after the batch
CLIENT_TIMEOUT_S = 60.0               # high, so a blocked ping is recorded not dropped
ZMQ_LINGER_MS = 1000
GPU_RELEASE_GAP_S = 5.0

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

OUT_PNG = os.path.join(os.getcwd(), "cb_hammer_results.png")
OUT_CSV = {
    "inprocess": os.path.join(os.getcwd(), "cb_hammer_inprocess.csv"),
    "zmq": os.path.join(os.getcwd(), "cb_hammer_zmq.csv"),
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
    fut = _futures.pop(rid, None)
    if fut is not None and not fut.done():
        if status_name == "IGNORED":
            fut.set_exception(RuntimeError("request IGNORED - KV cache OOM"))
        else:
            fut.set_result(_token_counts.pop(rid, 0))


# =========================================================================== #
# ZMQ ENGINE PROCESS                                                            #
# =========================================================================== #

def _handle_request(msg, pipe, handles) -> bool:
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
# BACKEND A - in-process engine                                                 #
# =========================================================================== #

class InProcessBackend:
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

        mpctx = mp.get_context("spawn")
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
# FASTAPI SERVER                                                                #
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
# CLIENT - concurrent hammer load                                               #
# =========================================================================== #

async def hammer_load_test(base_url: str) -> dict:
    """CONCURRENT_PINGERS clients hammer /ping continuously; a batch is fired
    partway through. Returns every (t_since_start, latency_ms) sample."""
    samples: list[tuple[float, float]] = []
    stop = asyncio.Event()
    limits = httpx.Limits(
        max_connections=CONCURRENT_PINGERS + BATCH_SIZE + 16,
        max_keepalive_connections=CONCURRENT_PINGERS + BATCH_SIZE + 16,
    )

    async with httpx.AsyncClient(base_url=base_url, timeout=CLIENT_TIMEOUT_S, limits=limits) as client:
        t_origin = time.perf_counter()

        async def pinger() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    await client.get("/ping")
                except Exception:  # noqa: BLE001
                    continue
                t1 = time.perf_counter()
                samples.append((t0 - t_origin, (t1 - t0) * 1000.0))

        pingers = [asyncio.create_task(pinger()) for _ in range(CONCURRENT_PINGERS)]

        # Phase 1: baseline - hammer only.
        print(f"[test]   phase 1: {CONCURRENT_PINGERS} pingers, no batch ({BASELINE_SECONDS:.0f}s) ...")
        await asyncio.sleep(BASELINE_SECONDS)

        # Phase 2: batch in flight - hammer continues.
        print(f"[test]   phase 2: firing {BATCH_SIZE} /generate, hammer continues ...")
        batch_start = time.perf_counter() - t_origin
        gen_task = asyncio.gather(*[client.post("/generate", params={"idx": i}) for i in range(BATCH_SIZE)])
        gen_results = await gen_task
        batch_end = time.perf_counter() - t_origin

        # Phase 3: cooldown - hammer only.
        print(f"[test]   phase 3: cooldown ({COOLDOWN_SECONDS:.0f}s) ...")
        await asyncio.sleep(COOLDOWN_SECONDS)

        stop.set()
        await asyncio.gather(*pingers, return_exceptions=True)

    return {
        "samples": samples,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "tokens": sum(r.json().get("tokens", 0) for r in gen_results),
    }


# --------------------------------------------------------------------------- #
# OUTPUT - CSV, summary table, graphs                                           #
# --------------------------------------------------------------------------- #

def _dump_csv(path: str, data: dict) -> None:
    with open(path, "w") as f:
        f.write("t_seconds,latency_ms\n")
        for t, lat in data["samples"]:
            f.write(f"{t:.6f},{lat:.4f}\n")
    print(f"[driver] raw samples written to {path}")


def _phase_pcts(data: dict) -> tuple[tuple, tuple]:
    import numpy as np

    if not data["samples"]:
        return (0.0, 0.0, 0.0, 0), (0.0, 0.0, 0.0, 0)
    arr = np.array(data["samples"])
    t, lat = arr[:, 0], arr[:, 1]
    base = lat[t < data["batch_start"]]
    batch = lat[(t >= data["batch_start"]) & (t <= data["batch_end"])]

    def pct(a):
        if len(a) == 0:
            return (0.0, 0.0, 0.0, 0)
        return (float(np.percentile(a, 50)), float(np.percentile(a, 99)), float(a.max()), len(a))

    return pct(base), pct(batch)


def print_summary(inproc: dict, zmqd: dict) -> None:
    in_base, in_batch = _phase_pcts(inproc)
    zq_base, zq_batch = _phase_pcts(zmqd)
    print("\n" + "=" * 78)
    print(f"SUMMARY - /ping under {CONCURRENT_PINGERS} concurrent clients")
    print("=" * 78)
    print(f"{'':26}{'IN-PROCESS':>24}{'OUT-OF-PROCESS (ZMQ)':>26}")
    fmt = lambda p: f"p50={p[0]:7.1f} p99={p[1]:8.1f} max={p[2]:8.1f}"
    print(f"{'baseline (no batch)':26}{fmt(in_base):>24}{fmt(zq_base):>26}")
    print(f"{'during batch':26}{fmt(in_batch):>24}{fmt(zq_batch):>26}")
    print(f"{'pings during batch':26}{in_batch[3]:>24}{zq_batch[3]:>26}")
    print("-" * 78)
    print("During-batch p50 is the honest number under concurrent load: a ping")
    print("cannot slip through a between-step gap when 48 are always in flight.")
    print("=" * 78)


def make_graphs(inproc: dict, zmqd: dict, png_path: str) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def as_arr(d):
        return np.array(d["samples"]) if d["samples"] else np.zeros((0, 2))

    a_in, a_zmq = as_arr(inproc), as_arr(zmqd)
    all_lat = np.concatenate([
        a_in[:, 1] if len(a_in) else np.array([1.0]),
        a_zmq[:, 1] if len(a_zmq) else np.array([1.0]),
    ])
    ymin = max(0.2, float(all_lat.min()) * 0.7)
    ymax = float(all_lat.max()) * 1.4

    fig, axes = plt.subplots(3, 1, figsize=(13, 16))

    def timeline(ax, data, title, p99_color):
        arr = as_arr(data)
        if len(arr):
            t, lat = arr[:, 0], arr[:, 1]
            # subsample raw points so the scatter stays legible
            if len(t) > 20000:
                idx = np.random.default_rng(0).choice(len(t), 20000, replace=False)
                ax.scatter(t[idx], lat[idx], s=3, alpha=0.12, color="#999999", linewidths=0)
            else:
                ax.scatter(t, lat, s=3, alpha=0.12, color="#999999", linewidths=0)
            # percentile lines, 100 ms bins
            edges = np.arange(0.0, float(t.max()) + 0.1, 0.1)
            centers, p50s, p99s = [], [], []
            for i in range(len(edges) - 1):
                mask = (t >= edges[i]) & (t < edges[i + 1])
                if not mask.any():
                    continue
                centers.append((edges[i] + edges[i + 1]) / 2)
                p50s.append(float(np.percentile(lat[mask], 50)))
                p99s.append(float(np.percentile(lat[mask], 99)))
            ax.plot(centers, p50s, color="#1f77b4", lw=1.6, label="p50 (100ms bins)")
            ax.plot(centers, p99s, color=p99_color, lw=1.6, label="p99 (100ms bins)")
        ax.axvspan(data["batch_start"], data["batch_end"],
                   color="#ffae42", alpha=0.18, label="batch in flight")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_xlabel("time since test start (s)")
        ax.set_ylabel("/ping latency (ms)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    timeline(axes[0], inproc,
             f"IN-PROCESS engine  -  /ping under {CONCURRENT_PINGERS} concurrent clients", "#d62728")
    timeline(axes[1], zmqd,
             f"OUT-OF-PROCESS (ZMQ) engine  -  /ping under {CONCURRENT_PINGERS} concurrent clients", "#2ca02c")

    # CDF of the during-batch samples
    ax = axes[2]
    for data, label, color in [(inproc, "in-process", "#d62728"),
                               (zmqd, "zmq out-of-process", "#2ca02c")]:
        arr = as_arr(data)
        if not len(arr):
            continue
        t, lat = arr[:, 0], arr[:, 1]
        mask = (t >= data["batch_start"]) & (t <= data["batch_end"])
        batch_lat = np.sort(lat[mask])
        if not len(batch_lat):
            continue
        y = np.arange(1, len(batch_lat) + 1) / len(batch_lat)
        ax.plot(batch_lat, y, color=color, lw=2.2, label=f"{label}  (n={len(batch_lat)})")
    ax.set_xscale("log")
    ax.set_title("/ping latency distribution DURING the batch window (CDF)")
    ax.set_xlabel("/ping latency (ms)")
    ax.set_ylabel("cumulative fraction of requests")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"[driver] graphs written to {png_path}")


# --------------------------------------------------------------------------- #
# DRIVER                                                                        #
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
        print(f"[driver] mode={mode} server ready, running hammer test ...")
        return asyncio.run(hammer_load_test(base_url))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_test() -> None:
    inproc = _run_one_mode("inprocess", PORT_INPROC)
    print(f"\n[driver] waiting {GPU_RELEASE_GAP_S:.0f}s for GPU memory to free ...")
    time.sleep(GPU_RELEASE_GAP_S)
    zmqd = _run_one_mode("zmq", PORT_ZMQ)

    _dump_csv(OUT_CSV["inprocess"], inproc)
    _dump_csv(OUT_CSV["zmq"], zmqd)
    print_summary(inproc, zmqd)
    make_graphs(inproc, zmqd, OUT_PNG)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if "--serve" in sys.argv:
        i = sys.argv.index("--serve")
        run_server(sys.argv[i + 1], int(sys.argv[i + 2]))
    else:
        run_test()