from __future__ import annotations

import argparse
import gc
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import openvino_genai as genai


REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class RequestState:
    request_id: int
    prompt: str
    input_tokens: int
    handle: genai.GenerationHandle
    generated_ids: list[int] = field(default_factory=list)
    first_token_s: float | None = None
    finish_s: float | None = None
    status: genai.GenerationStatus = genai.GenerationStatus.RUNNING
    finish_reason: genai.GenerationFinishReason = genai.GenerationFinishReason.NONE


@dataclass
class ScenarioResult:
    label: str
    duration_s: float
    total_input_tokens: int
    total_output_tokens: int
    finished: int
    ignored: int
    cancelled: int
    stopped: int
    cache_usage: float
    max_cache_usage: float
    avg_cache_usage: float
    requests_processed: int
    scheduled_requests: int
    samples: list[str]

    @property
    def output_tokens_per_s(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.total_output_tokens / self.duration_s


def resolve_sonnet_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        REPO_ROOT / "benchmark" / "sonnets.txt",
        REPO_ROOT / "benchmark" / "sonnet.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find benchmark/sonnets.txt or benchmark/sonnet.txt")


def count_tokens(tokenizer: genai.Tokenizer, prompt: str) -> int:
    encoded = tokenizer.encode(prompt)
    input_ids = encoded.input_ids
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    return len(input_ids)


def build_sonnet_prompts(
    tokenizer: genai.Tokenizer,
    sonnet_path: Path,
    num_requests: int,
    input_tokens: int,
    shared_prefix_tokens: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    lines = sonnet_path.read_text(encoding="utf-8").splitlines()
    lines = [line + "\n" for line in lines if line.strip()]
    if not lines:
        raise ValueError(f"Sonnet file is empty: {sonnet_path}")

    header = "Continue from these Shakespeare sonnet lines while preserving the style:\n"
    header_tokens = count_tokens(tokenizer, header)
    line_token_counts = [max(1, count_tokens(tokenizer, line)) for line in lines]
    avg_line_tokens = max(1.0, sum(line_token_counts) / len(line_token_counts))

    prefix_line_count = max(1, round(max(0, shared_prefix_tokens - header_tokens) / avg_line_tokens))
    total_line_count = max(prefix_line_count + 1, round(max(1, input_tokens - header_tokens) / avg_line_tokens))
    shared_prefix = lines[:prefix_line_count]
    extra_count = max(1, total_line_count - prefix_line_count)

    prompts = []
    for request_idx in range(num_requests):
        sampled = rng.choices(lines, k=extra_count)
        prompts.append(
            header
            + f"Request {request_idx}: finish the passage with a distinct final couplet.\n"
            + "".join(shared_prefix + sampled)
        )
    return prompts


def build_generation_config(output_tokens: int) -> genai.GenerationConfig:
    config = genai.GenerationConfig()
    config.max_new_tokens = output_tokens
    config.ignore_eos = True
    config.do_sample = False
    return config


def build_scheduler_config(args: argparse.Namespace, use_cache_eviction: bool) -> genai.SchedulerConfig:
    config = genai.SchedulerConfig()
    config.max_num_batched_tokens = args.max_num_batched_tokens
    config.max_num_seqs = args.max_num_seqs
    config.cache_size = args.cache_size
    if args.num_kv_blocks is not None:
        config.num_kv_blocks = args.num_kv_blocks
    config.dynamic_split_fuse = args.dynamic_split_fuse
    config.enable_prefix_caching = args.enable_prefix_caching
    config.use_cache_eviction = use_cache_eviction

    if use_cache_eviction:
        aggregation_mode = getattr(genai.AggregationMode, args.aggregation_mode)
        config.cache_eviction_config = genai.CacheEvictionConfig(
            args.eviction_start_size,
            args.eviction_recent_size,
            args.eviction_max_cache_size,
            aggregation_mode,
            args.apply_rotation,
            args.snapkv_window_size,
        )

    return config


def build_pipeline(args: argparse.Namespace, use_cache_eviction: bool) -> genai.ContinuousBatchingPipeline:
    return genai.ContinuousBatchingPipeline(
        args.model_dir,
        scheduler_config=build_scheduler_config(args, use_cache_eviction),
        device=args.device,
        properties=dict(args.property),
        tokenizer_properties={},
        vision_encoder_properties={},
    )


def submit_requests(
    pipeline: genai.ContinuousBatchingPipeline,
    prompts: list[str],
    generation_config: genai.GenerationConfig,
) -> dict[int, RequestState]:
    tokenizer = pipeline.get_tokenizer()
    active: dict[int, RequestState] = {}
    for request_id, prompt in enumerate(prompts):
        handle = pipeline.add_request(request_id, prompt, generation_config)
        active[request_id] = RequestState(
            request_id=request_id,
            prompt=prompt,
            input_tokens=count_tokens(tokenizer, prompt),
            handle=handle,
        )
    return active


def run_scenario(
    label: str,
    args: argparse.Namespace,
    prompts: list[str],
    generation_config: genai.GenerationConfig,
    use_cache_eviction: bool,
) -> ScenarioResult:
    print(f"\n=== {label} ===")
    pipeline = build_pipeline(args, use_cache_eviction=use_cache_eviction)
    tokenizer = pipeline.get_tokenizer()
    requests = submit_requests(pipeline, prompts, generation_config)

    start_s = time.perf_counter()
    while pipeline.has_non_finished_requests():
        pipeline.step()
        now_s = time.perf_counter()
        for state in requests.values():
            if state.handle.can_read():
                for output in state.handle.read().values():
                    if output.generated_ids:
                        if state.first_token_s is None:
                            state.first_token_s = now_s
                        state.generated_ids.extend(output.generated_ids)
                    if output.finish_reason != genai.GenerationFinishReason.NONE:
                        state.finish_reason = output.finish_reason

            status = state.handle.get_status()
            if status != genai.GenerationStatus.RUNNING and state.status == genai.GenerationStatus.RUNNING:
                state.status = status
                state.finish_s = now_s

    # Drain anything that became readable on the terminal step.
    end_s = time.perf_counter()
    for state in requests.values():
        if state.handle.can_read():
            for output in state.handle.read().values():
                if output.generated_ids:
                    if state.first_token_s is None:
                        state.first_token_s = end_s
                    state.generated_ids.extend(output.generated_ids)
                if output.finish_reason != genai.GenerationFinishReason.NONE:
                    state.finish_reason = output.finish_reason
        state.status = state.handle.get_status()
        if state.finish_s is None and state.status != genai.GenerationStatus.RUNNING:
            state.finish_s = end_s

    metrics = pipeline.get_metrics()
    samples = [
        tokenizer.decode(state.generated_ids[:120]).strip()
        for state in list(requests.values())[: args.samples]
        if state.generated_ids
    ]

    result = ScenarioResult(
        label=label,
        duration_s=end_s - start_s,
        total_input_tokens=sum(state.input_tokens for state in requests.values()),
        total_output_tokens=sum(len(state.generated_ids) for state in requests.values()),
        finished=sum(state.status == genai.GenerationStatus.FINISHED for state in requests.values()),
        ignored=sum(state.status == genai.GenerationStatus.IGNORED for state in requests.values()),
        cancelled=sum(state.status == genai.GenerationStatus.CANCEL for state in requests.values()),
        stopped=sum(state.status == genai.GenerationStatus.STOP for state in requests.values()),
        cache_usage=float(metrics.cache_usage),
        max_cache_usage=float(metrics.max_cache_usage),
        avg_cache_usage=float(metrics.avg_cache_usage),
        requests_processed=int(metrics.requests),
        scheduled_requests=int(metrics.scheduled_requests),
        samples=samples,
    )
    print_result(result)
    del pipeline
    gc.collect()
    return result


def print_result(result: ScenarioResult) -> None:
    print(f"Duration:            {result.duration_s:.2f} s")
    print(f"Finished requests:   {result.finished}")
    print(f"Ignored requests:    {result.ignored}")
    print(f"Cancelled requests:  {result.cancelled}")
    print(f"Stopped requests:    {result.stopped}")
    print(f"Input tokens:        {result.total_input_tokens}")
    print(f"Output tokens:       {result.total_output_tokens}")
    print(f"Output throughput:   {result.output_tokens_per_s:.2f} tok/s")
    print(f"Pipeline requests:   {result.requests_processed}")
    print(f"Scheduled requests:  {result.scheduled_requests}")
    print(f"Cache usage:         {result.cache_usage:.2f}%")
    print(f"Max cache usage:     {result.max_cache_usage:.2f}%")
    print(f"Avg cache usage:     {result.avg_cache_usage:.2f}%")


def print_comparison(no_eviction: ScenarioResult, eviction: ScenarioResult) -> None:
    print("\n=== Cache Eviction Utility ===")
    print(f"Finished delta:       {eviction.finished - no_eviction.finished:+d}")
    print(f"Ignored delta:        {eviction.ignored - no_eviction.ignored:+d}")
    print(f"Output token delta:   {eviction.total_output_tokens - no_eviction.total_output_tokens:+d}")
    print(f"Throughput delta:     {eviction.output_tokens_per_s - no_eviction.output_tokens_per_s:+.2f} tok/s")
    print(f"Max cache usage delta:{eviction.max_cache_usage - no_eviction.max_cache_usage:+.2f}%")

    if eviction.samples:
        print("\nSample decoded output with cache eviction:")
        print(eviction.samples[0][:800] or "<empty>")


def parse_property(value: str) -> tuple[str, str]:
    key, sep, prop_value = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("properties must be KEY=VALUE")
    return key, prop_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuous batching with and without KV token cache eviction "
            "using sonnet prompts under the same constrained cache budget."
        )
    )
    parser.add_argument("model_dir", help="Path to an OpenVINO GenAI model directory")
    parser.add_argument("--device", default="CPU", help="OpenVINO device string")
    parser.add_argument("--sonnet-path", default=None, help="Path to sonnets.txt/sonnet.txt")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=900)
    parser.add_argument("--shared-prefix-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)

    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--cache-size", type=int, default=1, help="KV cache size in GB")
    parser.add_argument("--num-kv-blocks", type=int, default=None)
    parser.add_argument("--dynamic-split-fuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eviction-start-size", type=int, default=128)
    parser.add_argument("--eviction-recent-size", type=int, default=384)
    parser.add_argument("--eviction-max-cache-size", type=int, default=1024)
    parser.add_argument(
        "--aggregation-mode",
        choices=sorted(genai.AggregationMode.__members__),
        default="NORM_SUM",
    )
    parser.add_argument("--apply-rotation", action="store_true")
    parser.add_argument("--snapkv-window-size", type=int, default=8)

    parser.add_argument(
        "--property",
        action="append",
        default=[],
        type=parse_property,
        metavar="KEY=VALUE",
        help="OpenVINO runtime property passed to ContinuousBatchingPipeline properties",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.property = dict(args.property)

    sonnet_path = resolve_sonnet_path(args.sonnet_path)
    tokenizer = genai.Tokenizer(args.model_dir)
    prompts = build_sonnet_prompts(
        tokenizer=tokenizer,
        sonnet_path=sonnet_path,
        num_requests=args.num_requests,
        input_tokens=args.input_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        seed=args.seed,
    )
    generation_config = build_generation_config(args.output_tokens)

    print(f"Model:        {args.model_dir}")
    print(f"Device:       {args.device}")
    print(f"Sonnet file:  {sonnet_path}")
    print(f"Requests:     {len(prompts)}")

    no_eviction = run_scenario(
        label="cache eviction OFF",
        args=args,
        prompts=prompts,
        generation_config=generation_config,
        use_cache_eviction=False,
    )
    eviction = run_scenario(
        label="cache eviction ON",
        args=args,
        prompts=prompts,
        generation_config=generation_config,
        use_cache_eviction=True,
    )
    print_comparison(no_eviction, eviction)


if __name__ == "__main__":
    main()
