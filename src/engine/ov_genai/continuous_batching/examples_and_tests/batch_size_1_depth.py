from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import openvino_genai as genai
from rich import box
from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TEXT_PATH = REPO_ROOT / "benchmark" / "sonnet.txt"
console = Console(width=180)


TERMINAL_STATUSES = {
    genai.GenerationStatus.FINISHED,
    genai.GenerationStatus.IGNORED,
    genai.GenerationStatus.CANCEL,
    genai.GenerationStatus.STOP,
}


@dataclass
class RunMetrics:
    run: int
    depth_tokens: int
    prompt_tokens: int
    max_new_tokens: int
    input_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_ms: float
    prefill_tps: float
    decode_tps: float
    decode_duration_s: float
    total_s: float
    cache_usage: float
    max_cache_usage: float
    avg_cache_usage: float
    kv_cache_gib: float
    status: str
    output_text: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_property(value: str) -> tuple[str, str]:
    key, sep, prop_value = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("properties must be KEY=VALUE")
    return key, prop_value


def count_tokens(tokenizer: genai.Tokenizer, prompt: str) -> int:
    input_ids = tokenizer.encode(prompt).input_ids
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and hasattr(input_ids[0], "__len__"):
        return len(input_ids[0])
    return len(input_ids)


def load_text_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"Text dataset is empty: {path}")
    return lines


def build_synthetic_prompt(
    tokenizer: genai.Tokenizer,
    text_lines: list[str],
    prompt_tokens: int,
    depth_tokens: int,
    seed: int,
) -> tuple[str, int]:
    """Concatenate small text snippets until the prompt reaches the target token count."""

    total_tokens = depth_tokens + prompt_tokens
    rng = random.Random(seed)
    sampled_lines = text_lines[:]
    rng.shuffle(sampled_lines)

    prompt_parts = [
        "Use the following synthetic context as source material, then continue it briefly.\n\n"
    ]
    prompt = "".join(prompt_parts)
    line_idx = 0
    current_tokens = count_tokens(tokenizer, prompt)

    while current_tokens < total_tokens:
        next_line = sampled_lines[line_idx % len(sampled_lines)] + "\n"
        prompt_parts.append(next_line)
        prompt += next_line
        line_idx += 1
        current_tokens = count_tokens(tokenizer, prompt)

    return prompt, current_tokens


def build_scheduler_config(args: argparse.Namespace) -> genai.SchedulerConfig:
    """Build a continuous batching scheduler that is intentionally batch size 1."""

    config = genai.SchedulerConfig()
    config.max_num_batched_tokens = args.max_num_batched_tokens
    config.max_num_seqs = 1
    config.cache_size = args.cache_size
    config.dynamic_split_fuse = args.dynamic_split_fuse
    config.enable_prefix_caching = args.enable_prefix_caching
    return config


def build_generation_config(args: argparse.Namespace) -> genai.GenerationConfig:
    config = genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.ignore_eos = args.ignore_eos
    config.do_sample = False
    return config


def build_pipeline(args: argparse.Namespace) -> genai.ContinuousBatchingPipeline:
    return genai.ContinuousBatchingPipeline(
        args.model_dir,
        device=args.device,
        scheduler_config=build_scheduler_config(args),
        properties=dict(args.property),
        tokenizer_properties={},
        vision_encoder_properties={},
    )


def get_cache_metrics(pipeline: genai.ContinuousBatchingPipeline) -> tuple[float, float, float, float]:
    metrics = pipeline.get_metrics()
    kv_cache_bytes = float(getattr(metrics, "kv_cache_size_in_bytes", 0.0))
    return (
        float(getattr(metrics, "cache_usage", 0.0)),
        float(getattr(metrics, "max_cache_usage", 0.0)),
        float(getattr(metrics, "avg_cache_usage", 0.0)),
        kv_cache_bytes / (1024.0**3),
    )


def run_once(
    pipeline: genai.ContinuousBatchingPipeline,
    tokenizer: genai.Tokenizer,
    generation_config: genai.GenerationConfig,
    prompt: str,
    input_tokens: int,
    run: int,
    args: argparse.Namespace,
) -> RunMetrics:
    start_s = time.perf_counter()
    handle = pipeline.add_request(run, prompt, generation_config)

    output_tokens = 0
    output_token_ids: list[int] = []
    first_token_s: float | None = None
    last_token_s: float | None = None
    status = genai.GenerationStatus.RUNNING

    while True:
        pipeline.step()
        now_s = time.perf_counter()

        if handle.can_read():
            for output in handle.read().values():
                generated_ids = output.generated_ids
                new_tokens = len(generated_ids) if generated_ids is not None else 0
                if new_tokens:
                    if first_token_s is None:
                        first_token_s = now_s
                    last_token_s = now_s
                    output_tokens += new_tokens
                    output_token_ids.extend(generated_ids)

        status = handle.get_status()
        if status in TERMINAL_STATUSES:
            break

    end_s = time.perf_counter()
    cache_usage, max_cache_usage, avg_cache_usage, kv_cache_gib = get_cache_metrics(pipeline)
    if first_token_s is None:
        first_token_s = end_s
    if last_token_s is None:
        last_token_s = first_token_s

    ttft_s = first_token_s - start_s
    decode_duration_s = max(last_token_s - first_token_s, 0.0)
    decode_tokens = max(output_tokens - 1, 0)
    output_text = tokenizer.decode(output_token_ids).strip() if output_token_ids else ""

    return RunMetrics(
        run=run,
        depth_tokens=args.depth,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_s=ttft_s,
        tpot_ms=(decode_duration_s * 1000.0 / decode_tokens) if decode_tokens else 0.0,
        prefill_tps=(input_tokens / ttft_s) if ttft_s > 0 else 0.0,
        decode_tps=(decode_tokens / decode_duration_s) if decode_duration_s > 0 else 0.0,
        decode_duration_s=decode_duration_s,
        total_s=end_s - start_s,
        cache_usage=cache_usage,
        max_cache_usage=max_cache_usage,
        avg_cache_usage=avg_cache_usage,
        kv_cache_gib=kv_cache_gib,
        status=status.name,
        output_text=output_text,
    )


def average_metrics(runs: list[RunMetrics]) -> RunMetrics:
    def mean(name: str) -> float:
        return statistics.fmean(getattr(run, name) for run in runs)

    return RunMetrics(
        run=0,
        depth_tokens=runs[0].depth_tokens,
        prompt_tokens=runs[0].prompt_tokens,
        max_new_tokens=runs[0].max_new_tokens,
        input_tokens=round(mean("input_tokens")),
        output_tokens=round(mean("output_tokens")),
        ttft_s=mean("ttft_s"),
        tpot_ms=mean("tpot_ms"),
        prefill_tps=mean("prefill_tps"),
        decode_tps=mean("decode_tps"),
        decode_duration_s=mean("decode_duration_s"),
        total_s=mean("total_s"),
        cache_usage=mean("cache_usage"),
        max_cache_usage=mean("max_cache_usage"),
        avg_cache_usage=mean("avg_cache_usage"),
        kv_cache_gib=mean("kv_cache_gib"),
        status="",
        output_text="",
    )


def metric_rows(label: str, metrics: RunMetrics) -> list[tuple[str, str]]:
    prefix = f"{label} "
    return [
        (prefix + "d/p/n", f"{metrics.depth_tokens}/{metrics.prompt_tokens}/{metrics.max_new_tokens}"),
        (prefix + "input tokens", str(metrics.input_tokens)),
        (prefix + "output tokens", str(metrics.output_tokens)),
        (prefix + "ttft", f"{metrics.ttft_s:.4f} s"),
        (prefix + "tpot", f"{metrics.tpot_ms:.2f} ms/token"),
        (prefix + "prefill", f"{metrics.prefill_tps:.1f} tokens/s"),
        (prefix + "decode", f"{metrics.decode_tps:.1f} tokens/s"),
        (prefix + "decode time", f"{metrics.decode_duration_s:.4f} s"),
        (prefix + "total time", f"{metrics.total_s:.4f} s"),
        (prefix + "cache usage", f"{metrics.cache_usage:.2f}%"),
        (prefix + "max cache usage", f"{metrics.max_cache_usage:.2f}%"),
        (prefix + "avg cache usage", f"{metrics.avg_cache_usage:.2f}%"),
        (prefix + "kv cache", f"{metrics.kv_cache_gib:.2f} GiB"),
        (prefix + "status", metrics.status or "-"),
    ]


def print_metrics_table(runs: list[RunMetrics]) -> None:
    table = Table(title="Batch Size 1 Depth Evaluation", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right", no_wrap=True)

    rows: list[tuple[str, str]] = []
    for metrics in runs:
        if rows:
            rows.append(("", ""))
        rows.extend(metric_rows(f"run {metrics.run}", metrics))

    if len(runs) > 1:
        rows.append(("", ""))
        rows.extend(metric_rows("avg", average_metrics(runs)))

    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)


def print_outputs_table(runs: list[RunMetrics]) -> None:
    table = Table(title="Generated Output", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("run", style="cyan", justify="right", no_wrap=True)
    table.add_column("output", overflow="fold")

    for metrics in runs:
        table.add_row(str(metrics.run), metrics.output_text or "<no decoded output>")
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OpenVINO GenAI ContinuousBatchingPipeline at batch size 1 "
            "with configurable synthetic context depth."
        )
    )
    parser.add_argument("model_dir", help="Path to an OpenVINO GenAI model directory")
    parser.add_argument("--device", default="GPU.0", help="OpenVINO device string")
    parser.add_argument(
        "--text-path",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="Small text dataset to concatenate into synthetic prompts",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=non_negative_int,
        default=0,
        help="Synthetic prior context tokens prepended before the p-token segment",
    )
    parser.add_argument(
        "--prompt-tokens",
        "--p",
        type=positive_int,
        default=1,
        help="Measured prompt segment length after depth tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--n",
        type=positive_int,
        default=128,
        help="Maximum generated tokens",
    )
    parser.add_argument("--runs", "-r", type=positive_int, default=1, help="Repeated runs")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for shuffled text lines")

    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=2048)
    parser.add_argument("--cache-size", type=positive_int, default=14, help="KV cache size in GB")
    parser.add_argument("--dynamic-split-fuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--property",
        action="append",
        default=["KV_CACHE_PRECISION"],
        type=parse_property,
        metavar="KEY=VALUE",
        help="OpenVINO runtime property passed to ContinuousBatchingPipeline properties",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Batch size: 1")
    print(f"Model:      {args.model_dir}")
    print(f"Device:     {args.device}")
    print(f"Text:       {args.text_path}")
    print(f"d/p/n:      {args.depth}/{args.prompt_tokens}/{args.max_new_tokens}")
    print(f"Runs:       {args.runs}\n")

    pipeline = build_pipeline(args)
    generation_config = build_generation_config(args)
    tokenizer = pipeline.get_tokenizer()
    text_lines = load_text_lines(args.text_path)

    metrics = []
    for run in range(1, args.runs + 1):
        prompt, input_tokens = build_synthetic_prompt(
            tokenizer=tokenizer,
            text_lines=text_lines,
            prompt_tokens=args.prompt_tokens,
            depth_tokens=args.depth,
            seed=args.seed + run - 1,
        )
        result = run_once(
            pipeline=pipeline,
            tokenizer=tokenizer,
            generation_config=generation_config,
            prompt=prompt,
            input_tokens=input_tokens,
            run=run,
            args=args,
        )
        metrics.append(result)

    print_metrics_table(metrics)
    print_outputs_table(metrics)


if __name__ == "__main__":
    main()
