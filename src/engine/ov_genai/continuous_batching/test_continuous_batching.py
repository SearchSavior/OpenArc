import random
import time
import statistics
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from openvino_genai import (
    GenerationConfig,
    ContinuousBatchingPipeline,
    SchedulerConfig,
    Tokenizer,
    GenerationFinishReason,
)


# ── parameters ─────────────────────────────────────────────────────────────────

#MODEL_DIR    = ""
MODEL_DIR  = "/mnt/Ironwolf-4TB/Models/OpenVINO/Deepseek/DeepSeek-R1-0528-Qwen3-8B-OpenVINO/DeepSeek-R1-0528-Qwen3-8B-int8_asym-ov/"
DEVICE       = "HETERO:GPU.0,GPU.1"
SONNET_PATH  = Path(__file__).parent / "sonnet.txt"

NUM_REQUESTS = 72 # set of all requests to be processed in one step
SEED         = 0

# Sonnet benchmark
INPUT_LEN    = 550   # target input tokens per request
OUTPUT_LEN   = 1024   # fixed output tokens per request (ignore_eos=True)
PREFIX_LEN   = 200   # shared prefix tokens (same across all requests)

# Scheduler
MAX_BATCHED_TOKENS  = 2048   # max tokens processed in one scheduler step (prompt + decode combined)
MAX_SEQS            = 16     # max sequences held in-flight simultaneously
CACHE_SIZE_GB       = 10     # KV cache size allocated on device (GB)
DYNAMIC_SPLIT_FUSE  = True   # split long prefills across steps to keep decode latency smooth
ENABLE_PREFIX_CACHE = True   # reuse KV blocks for shared prompt prefixes across requests
GRAPH_OUTPUT_DIR    = Path(__file__).parent / "bench_graphs"
SQLITE_DB_PATH      = Path(__file__).parent / "bench_results.sqlite"

# ── pipeline init ───────────────────────────────────────────────────────────────

genai_tokenizer = Tokenizer(str(MODEL_DIR))

scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = MAX_BATCHED_TOKENS
scheduler_config.max_num_seqs          = MAX_SEQS
scheduler_config.cache_size            = CACHE_SIZE_GB
scheduler_config.dynamic_split_fuse   = DYNAMIC_SPLIT_FUSE
scheduler_config.enable_prefix_caching = ENABLE_PREFIX_CACHE

pipeline = ContinuousBatchingPipeline(
    MODEL_DIR,
    device=DEVICE,
    scheduler_config=scheduler_config,
    properties={
        "MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"
    }
    #tokenizer=genai_tokenizer,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _count_input_tokens(tokenizer, prompt: str) -> int:
    try:
        ids = tokenizer.encode(prompt).input_ids
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        if ids and hasattr(ids[0], "__len__"):
            return len(ids[0])
        return len(ids)
    except Exception:
        return 0


# ── vLLMSonnetBench ────────────────────────────────────────────────────────────

class vLLMSonnetBench:
    """
    Sonnet-style benchmark prompt generator, modelled on vLLM's SonnetDataset.

    Loads a plain-text file of poem lines (e.g. sonnet.txt from the vLLM repo).
    Each request is assembled from:
      - a fixed shared prefix block (first N lines, same for every request) —
        intentionally exercises the prefix-cache path when prefix caching is on
      - randomly sampled extra lines drawn with replacement to reach `input_len`

    Output length is fixed at `output_len` tokens and enforced via
    `ignore_eos=True` so every request generates exactly that many tokens,
    making throughput numbers directly comparable across runs.
    """

    HEADER = "Pick as many lines as you can from these poem lines:\n"

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: Tokenizer,
        input_len:  int = 550,
        output_len: int = 150,
        prefix_len: int = 200,
        seed:       int = 0,
    ) -> None:
        self.tokenizer  = tokenizer
        self.input_len  = input_len
        self.output_len = output_len
        self.prefix_len = prefix_len
        random.seed(seed)

        with open(dataset_path, encoding="utf-8") as f:
            self.lines = f.readlines()
        if not self.lines:
            raise ValueError(f"Sonnet file is empty: {dataset_path}")

        # Average tokens per poem line (used to estimate how many lines to pick)
        line_lens   = [_count_input_tokens(tokenizer, ln) for ln in self.lines]
        self.avg_len = sum(line_lens) / len(line_lens)

        header_len = _count_input_tokens(tokenizer, self.HEADER)
        if input_len <= header_len:
            raise ValueError(
                f"input_len ({input_len}) must be greater than the header "
                f"token count ({header_len})."
            )

        # How many lines the full prompt should contain
        self._num_input_lines  = max(1, round((input_len  - header_len) / self.avg_len))
        # How many of those are the fixed shared prefix
        self._num_prefix_lines = max(0, round((prefix_len - header_len) / self.avg_len))
        self._prefix_lines     = self.lines[: self._num_prefix_lines]

        print(
            f"[SonnetBench] avg line len: {self.avg_len:.1f} tok | "
            f"prefix lines: {self._num_prefix_lines} | "
            f"total lines/request: {self._num_input_lines} | "
            f"target output: {output_len} tok"
        )

    def sample(self, num_requests: int) -> tuple[list[str], GenerationConfig]:
        """
        Return `(prompts, generation_config)` ready to pass to the pipeline.

        The generation config fixes output length exactly (`ignore_eos=True`)
        and disables sampling so results are deterministic given the seed.
        """
        n_extra  = max(0, self._num_input_lines - self._num_prefix_lines)
        prompts  = []
        for _ in range(num_requests):
            extra  = random.choices(self.lines, k=n_extra)
            body   = "".join(self._prefix_lines + extra)
            prompts.append(self.HEADER + body)

        gen_cfg = GenerationConfig(
            max_new_tokens=self.output_len,
            ignore_eos=True,          # enforce fixed output length
            do_sample=False,          # greedy — removes sampling noise
        )
        return prompts, gen_cfg


bench = vLLMSonnetBench(
    dataset_path=SONNET_PATH,
    tokenizer=genai_tokenizer,
    input_len=INPUT_LEN,
    output_len=OUTPUT_LEN,
    prefix_len=PREFIX_LEN,
    seed=SEED,
)
prompts, generation_config = bench.sample(NUM_REQUESTS)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _report_row(label: str, value: str, width: int = 50) -> str:
    gap = width - len(label) - len(value)
    return label + " " * max(gap, 1) + value


def _section(title: str, char: str, width: int = 50) -> str:
    return title.center(width, char)


class vLLMBenchData:
    """Owns request/step data collection and aggregate metric computation."""

    def __init__(self, tokenizer: Tokenizer, prompts: list[str]) -> None:
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.num_prompts = len(prompts)

        self.arrival_time: dict[int, float] = {}
        self.first_tok_time: dict[int, float] = {}
        self.finish_time: dict[int, float] = {}
        self.tok_timestamps: dict[int, list[float]] = {}
        self.tok_count: dict[int, int] = {}
        self.tok_ids: dict[int, list[int]] = {}
        self.input_tok_count: dict[int, int] = {}
        self.failed_ids: set[int] = set()
        self.finished_ids: set[int] = set()
        self.step_records: list[tuple[float, int, int]] = []

        self.benchmark_start = 0.0
        self.benchmark_end = 0.0
        self.benchmark_duration = 0.0

        for i, prompt in enumerate(prompts):
            self.input_tok_count[i] = _count_input_tokens(tokenizer, prompt)
            self.tok_timestamps[i] = []
            self.tok_count[i] = 0
            self.tok_ids[i] = []

    def start(self) -> None:
        self.benchmark_start = time.perf_counter()

    def stop(self) -> None:
        self.benchmark_end = time.perf_counter()
        self.benchmark_duration = self.benchmark_end - self.benchmark_start

    def mark_arrival(self, req_id: int) -> None:
        self.arrival_time[req_id] = time.perf_counter()

    def consume_output(self, req_id: int, output, step_ts: float) -> int:
        n_new = len(output.generated_ids) if output.generated_ids else 0
        if n_new > 0:
            if req_id not in self.first_tok_time:
                self.first_tok_time[req_id] = step_ts
            self.tok_timestamps[req_id].extend([step_ts] * n_new)
            self.tok_count[req_id] += n_new
            self.tok_ids[req_id].extend(output.generated_ids)

        if output.finish_reason not in (GenerationFinishReason.NONE, None):
            self.finished_ids.add(req_id)
            self.finish_time[req_id] = step_ts

        return n_new

    def add_step(self, step_ts: float, concurrent_now: int, new_tokens: int) -> None:
        self.step_records.append((step_ts, concurrent_now, new_tokens))

    def aggregate(self) -> dict[str, object]:
        ttfts: list[float] = []
        tpots: list[float] = []
        itls: list[float] = []

        for i in range(self.num_prompts):
            toks = self.tok_timestamps[i]
            n = len(toks)

            if i in self.first_tok_time:
                ttfts.append((self.first_tok_time[i] - self.arrival_time[i]) * 1000)

            if n > 1:
                # TPOT: average step-to-step interval after the first token
                tpots.append((toks[-1] - toks[0]) * 1000 / (n - 1))

                # ITL: drop zero-duration pairs from the same scheduler step.
                for j in range(1, n):
                    dt_ms = (toks[j] - toks[j - 1]) * 1000
                    if dt_ms > 0:
                        itls.append(dt_ms)

        total_input_tokens = sum(self.input_tok_count.values())
        total_output_tokens = sum(self.tok_count.values())
        successful = self.num_prompts - len(self.failed_ids)

        req_throughput = successful / self.benchmark_duration
        out_tok_throughput = total_output_tokens / self.benchmark_duration
        tot_tok_throughput = (
            total_input_tokens + total_output_tokens
        ) / self.benchmark_duration

        peak_concurrent = max((r[1] for r in self.step_records), default=0)

        all_tok_events = sorted(ts for tss in self.tok_timestamps.values() for ts in tss)
        peak_out_tok_tp = 0.0
        if all_tok_events:
            n_ev, j = len(all_tok_events), 0
            for k, t0 in enumerate(all_tok_events):
                while j < n_ev and all_tok_events[j] - t0 <= 1.0:
                    j += 1
                peak_out_tok_tp = max(peak_out_tok_tp, float(j - k))

        return {
            "ttfts": ttfts,
            "tpots": tpots,
            "itls": itls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "successful": successful,
            "req_throughput": req_throughput,
            "out_tok_throughput": out_tok_throughput,
            "tot_tok_throughput": tot_tok_throughput,
            "peak_concurrent": peak_concurrent,
            "peak_out_tok_tp": peak_out_tok_tp,
            "benchmark_duration": self.benchmark_duration,
        }

    def latency_summary(self, summary: dict[str, object]) -> dict[str, float]:
        ttfts = summary["ttfts"]
        tpots = summary["tpots"]
        itls = summary["itls"]
        return {
            "mean_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
            "median_ttft_ms": statistics.median(ttfts) if ttfts else 0.0,
            "p99_ttft_ms": percentile(ttfts, 99),
            "mean_tpot_ms": statistics.mean(tpots) if tpots else 0.0,
            "median_tpot_ms": statistics.median(tpots) if tpots else 0.0,
            "p99_tpot_ms": percentile(tpots, 99),
            "mean_itl_ms": statistics.mean(itls) if itls else 0.0,
            "median_itl_ms": statistics.median(itls) if itls else 0.0,
            "p99_itl_ms": percentile(itls, 99),
        }

    @staticmethod
    def print_pipeline_metrics(metrics) -> None:
        print("\nPipeline system metrics:")
        print(f"  Requests processed:    {metrics.requests}")
        print(f"  Scheduled requests:    {metrics.scheduled_requests}")
        print(f"  Cache usage:           {metrics.cache_usage:.2f}%")
        print(f"  Max cache usage:       {metrics.max_cache_usage:.2f}%")
        print(f"  Average cache usage:   {metrics.avg_cache_usage:.2f}%")

    def save_graphs(self, summary: dict[str, object], out_dir: str | Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("\nGraph generation skipped: matplotlib is not available.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1) Throughput timeline (tokens/s per scheduler step)
        x_sec: list[float] = []
        y_tok_s: list[float] = []
        prev_ts = self.benchmark_start
        for step_ts, _concurrent, new_tokens in self.step_records:
            dt = step_ts - prev_ts
            prev_ts = step_ts
            if dt <= 0:
                continue
            x_sec.append(step_ts - self.benchmark_start)
            y_tok_s.append(new_tokens / dt)

        if x_sec and y_tok_s:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_sec, y_tok_s, linewidth=1.5)
            ax.set_title("Output Throughput Timeline")
            ax.set_xlabel("Time Since Start (s)")
            ax.set_ylabel("Tokens/s")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_path / "throughput_timeline.png", dpi=140)
            plt.close(fig)

        # 2) TTFT histogram
        ttfts = summary["ttfts"]
        if ttfts:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(ttfts, bins=min(30, max(5, len(ttfts) // 2)), alpha=0.85)
            ax.set_title("TTFT Distribution")
            ax.set_xlabel("TTFT (ms)")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_path / "ttft_hist.png", dpi=140)
            plt.close(fig)

        # 3) TPOT histogram
        tpots = summary["tpots"]
        if tpots:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(tpots, bins=min(30, max(5, len(tpots) // 2)), alpha=0.85)
            ax.set_title("TPOT Distribution")
            ax.set_xlabel("TPOT (ms)")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_path / "tpot_hist.png", dpi=140)
            plt.close(fig)

        # 4) End-to-end latency scatter by request id
        req_ids: list[int] = []
        e2e_ms: list[float] = []
        out_toks: list[int] = []
        for i in range(self.num_prompts):
            if i in self.arrival_time and i in self.finish_time:
                req_ids.append(i)
                e2e_ms.append((self.finish_time[i] - self.arrival_time[i]) * 1000)
                out_toks.append(self.tok_count[i])
        if req_ids:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(req_ids, e2e_ms, s=20, alpha=0.8, c=out_toks, cmap="viridis")
            ax.set_title("Request End-to-End Latency")
            ax.set_xlabel("Request ID")
            ax.set_ylabel("Latency (ms)")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path / "request_e2e_scatter.png", dpi=140)
            plt.close(fig)

        print(f"\nSaved graphs to: {out_path}")

    def save_sqlite(
        self,
        summary: dict[str, object],
        latency: dict[str, float],
        metrics,
        db_path: str | Path,
        run_meta: dict[str, object],
    ) -> int:
        db = sqlite3.connect(str(db_path))
        cur = db.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                model_dir TEXT NOT NULL,
                device TEXT NOT NULL,
                num_prompts INTEGER NOT NULL,
                input_len INTEGER NOT NULL,
                output_len INTEGER NOT NULL,
                prefix_len INTEGER NOT NULL,
                max_batched_tokens INTEGER NOT NULL,
                max_seqs INTEGER NOT NULL,
                cache_size_gb REAL NOT NULL,
                dynamic_split_fuse INTEGER NOT NULL,
                enable_prefix_cache INTEGER NOT NULL,
                benchmark_duration_s REAL NOT NULL,
                successful_requests INTEGER NOT NULL,
                failed_requests INTEGER NOT NULL,
                total_input_tokens INTEGER NOT NULL,
                total_output_tokens INTEGER NOT NULL,
                req_throughput REAL NOT NULL,
                out_tok_throughput REAL NOT NULL,
                total_tok_throughput REAL NOT NULL,
                peak_output_tok_throughput REAL NOT NULL,
                peak_concurrent_requests REAL NOT NULL,
                mean_ttft_ms REAL NOT NULL,
                median_ttft_ms REAL NOT NULL,
                p99_ttft_ms REAL NOT NULL,
                mean_tpot_ms REAL NOT NULL,
                median_tpot_ms REAL NOT NULL,
                p99_tpot_ms REAL NOT NULL,
                mean_itl_ms REAL NOT NULL,
                median_itl_ms REAL NOT NULL,
                p99_itl_ms REAL NOT NULL,
                pipeline_requests INTEGER NOT NULL,
                pipeline_scheduled_requests INTEGER NOT NULL,
                pipeline_cache_usage REAL NOT NULL,
                pipeline_max_cache_usage REAL NOT NULL,
                pipeline_avg_cache_usage REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS request_metrics (
                run_id INTEGER NOT NULL,
                request_id INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                arrival_s REAL,
                first_token_s REAL,
                finish_s REAL,
                ttft_ms REAL,
                tpot_ms REAL,
                e2e_ms REAL,
                status TEXT NOT NULL,
                PRIMARY KEY (run_id, request_id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS step_metrics (
                run_id INTEGER NOT NULL,
                step_idx INTEGER NOT NULL,
                t_rel_s REAL NOT NULL,
                concurrent_requests INTEGER NOT NULL,
                new_tokens INTEGER NOT NULL,
                PRIMARY KEY (run_id, step_idx),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )

        created_at_utc = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO runs (
                created_at_utc, model_dir, device, num_prompts, input_len, output_len, prefix_len,
                max_batched_tokens, max_seqs, cache_size_gb, dynamic_split_fuse, enable_prefix_cache,
                benchmark_duration_s, successful_requests, failed_requests, total_input_tokens,
                total_output_tokens, req_throughput, out_tok_throughput, total_tok_throughput,
                peak_output_tok_throughput, peak_concurrent_requests, mean_ttft_ms, median_ttft_ms,
                p99_ttft_ms, mean_tpot_ms, median_tpot_ms, p99_tpot_ms, mean_itl_ms, median_itl_ms,
                p99_itl_ms, pipeline_requests, pipeline_scheduled_requests, pipeline_cache_usage,
                pipeline_max_cache_usage, pipeline_avg_cache_usage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_utc,
                str(run_meta["model_dir"]),
                str(run_meta["device"]),
                int(run_meta["num_prompts"]),
                int(run_meta["input_len"]),
                int(run_meta["output_len"]),
                int(run_meta["prefix_len"]),
                int(run_meta["max_batched_tokens"]),
                int(run_meta["max_seqs"]),
                float(run_meta["cache_size_gb"]),
                int(bool(run_meta["dynamic_split_fuse"])),
                int(bool(run_meta["enable_prefix_cache"])),
                float(summary["benchmark_duration"]),
                int(summary["successful"]),
                int(self.num_prompts - summary["successful"]),
                int(summary["total_input_tokens"]),
                int(summary["total_output_tokens"]),
                float(summary["req_throughput"]),
                float(summary["out_tok_throughput"]),
                float(summary["tot_tok_throughput"]),
                float(summary["peak_out_tok_tp"]),
                float(summary["peak_concurrent"]),
                float(latency["mean_ttft_ms"]),
                float(latency["median_ttft_ms"]),
                float(latency["p99_ttft_ms"]),
                float(latency["mean_tpot_ms"]),
                float(latency["median_tpot_ms"]),
                float(latency["p99_tpot_ms"]),
                float(latency["mean_itl_ms"]),
                float(latency["median_itl_ms"]),
                float(latency["p99_itl_ms"]),
                int(metrics.requests),
                int(metrics.scheduled_requests),
                float(metrics.cache_usage),
                float(metrics.max_cache_usage),
                float(metrics.avg_cache_usage),
            ),
        )
        run_id = int(cur.lastrowid)

        req_rows = []
        for i in range(self.num_prompts):
            arrival = self.arrival_time.get(i)
            first = self.first_tok_time.get(i)
            finish = self.finish_time.get(i)
            n = len(self.tok_timestamps[i])

            ttft_ms = (first - arrival) * 1000 if arrival is not None and first is not None else None
            tpot_ms = None
            if n > 1:
                tpot_ms = (self.tok_timestamps[i][-1] - self.tok_timestamps[i][0]) * 1000 / (n - 1)
            e2e_ms = (finish - arrival) * 1000 if arrival is not None and finish is not None else None

            status = "finished" if i in self.finished_ids else "unfinished"
            req_rows.append(
                (
                    run_id,
                    i,
                    int(self.input_tok_count[i]),
                    int(self.tok_count[i]),
                    arrival,
                    first,
                    finish,
                    ttft_ms,
                    tpot_ms,
                    e2e_ms,
                    status,
                )
            )

        cur.executemany(
            """
            INSERT INTO request_metrics (
                run_id, request_id, input_tokens, output_tokens, arrival_s, first_token_s, finish_s,
                ttft_ms, tpot_ms, e2e_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            req_rows,
        )

        step_rows = []
        for step_idx, (step_ts, concurrent, new_tokens) in enumerate(self.step_records):
            step_rows.append(
                (
                    run_id,
                    step_idx,
                    float(step_ts - self.benchmark_start),
                    int(concurrent),
                    int(new_tokens),
                )
            )
        cur.executemany(
            """
            INSERT INTO step_metrics (
                run_id, step_idx, t_rel_s, concurrent_requests, new_tokens
            ) VALUES (?, ?, ?, ?, ?)
            """,
            step_rows,
        )

        db.commit()
        db.close()
        return run_id


# ── benchmark ──────────────────────────────────────────────────────────────────

num_prompts = len(prompts)
print(f"Starting benchmark with {num_prompts} prompts…")
print("-" * 50)

bench_data = vLLMBenchData(genai_tokenizer, prompts)
bench_data.start()

# Submit every request before the step loop so the scheduler sees the full queue
handles = []
for i, prompt in enumerate(prompts):
    bench_data.mark_arrival(i)
    handle = pipeline.add_request(i, prompt, generation_config)
    handles.append(handle)

# Step loop — read token-level output after each scheduler step
while pipeline.has_non_finished_requests():
    concurrent_now = num_prompts - len(bench_data.finished_ids)
    pipeline.step()
    step_ts = time.perf_counter()
    step_new_toks = 0

    for i, handle in enumerate(handles):
        if i in bench_data.finished_ids or not handle.can_read():
            continue

        # read() returns {seq_idx: GenerationOutput} with the tokens from this step
        for output in handle.read().values():
            step_new_toks += bench_data.consume_output(i, output, step_ts)

    bench_data.add_step(step_ts, concurrent_now, step_new_toks)

bench_data.stop()
summary = bench_data.aggregate()
lat = bench_data.latency_summary(summary)

# ── report ─────────────────────────────────────────────────────────────────────

W = 50

def _row(label: str, value) -> None:
    if isinstance(value, float):
        vs = f"{value:.2f}"
    else:
        vs = str(value)
    print(_report_row(label, vs, W))

print()
print(_section(" Serving Benchmark Result ", "=", W))
_row("Successful requests:",          summary["successful"])
_row("Failed requests:",              len(bench_data.failed_ids))
_row("Benchmark duration (s):",       summary["benchmark_duration"])
_row("Total input tokens:",           summary["total_input_tokens"])
_row("Total generated tokens:",       summary["total_output_tokens"])
_row("Request throughput (req/s):",   summary["req_throughput"])
_row("Output token throughput (tok/s):", summary["out_tok_throughput"])
_row("Peak output token throughput (tok/s):", summary["peak_out_tok_tp"])
_row("Peak concurrent requests:",     float(summary["peak_concurrent"]))
_row("Total token throughput (tok/s):", summary["tot_tok_throughput"])
print(_section("Time to First Token", "-", W))
_row("Mean TTFT (ms):",   lat["mean_ttft_ms"])
_row("Median TTFT (ms):", lat["median_ttft_ms"])
_row("P99 TTFT (ms):",    lat["p99_ttft_ms"])
print(_section("Time per Output Token (excl. 1st token)", "-", W))
_row("Mean TPOT (ms):",   lat["mean_tpot_ms"])
_row("Median TPOT (ms):", lat["median_tpot_ms"])
_row("P99 TPOT (ms):",    lat["p99_tpot_ms"])
print(_section("Inter-token Latency", "-", W))
_row("Mean ITL (ms):",    lat["mean_itl_ms"])
_row("Median ITL (ms):",  lat["median_itl_ms"])
_row("P99 ITL (ms):",     lat["p99_itl_ms"])
print("=" * W)

# ── pipeline system metrics ────────────────────────────────────────────────────

metrics = pipeline.get_metrics()
bench_data.print_pipeline_metrics(metrics)
bench_data.save_graphs(summary, GRAPH_OUTPUT_DIR)
run_meta = {
    "model_dir": MODEL_DIR,
    "device": DEVICE,
    "num_prompts": NUM_REQUESTS,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "prefix_len": PREFIX_LEN,
    "max_batched_tokens": MAX_BATCHED_TOKENS,
    "max_seqs": MAX_SEQS,
    "cache_size_gb": CACHE_SIZE_GB,
    "dynamic_split_fuse": DYNAMIC_SPLIT_FUSE,
    "enable_prefix_cache": ENABLE_PREFIX_CACHE,
}
run_id = bench_data.save_sqlite(summary, lat, metrics, SQLITE_DB_PATH, run_meta)
print(f"Saved benchmark to SQLite: {SQLITE_DB_PATH} (run_id={run_id})")

# ── sample outputs ─────────────────────────────────────────────────────────────

print("\nSample outputs (first 5 requests):")
print("=" * W)
for i in range(min(5, num_prompts)):
    text = genai_tokenizer.decode(bench_data.tok_ids[i])
    print(f"[{i}] prompt : {prompts[i][:80].strip()!r}")
    print(f"[{i}] output : {text[:200].strip()!r}")
    print("-" * W)
