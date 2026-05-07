from __future__ import annotations

import time

import openvino_genai as genai

MODEL_DIR = "/mnt/Ironwolf-4TB/Models/OpenVINO/Gemma/gemma-3-4b-it-int4_asym-ov/"
DEVICE = "GPU.0"
PROMPT = (
    "Explain in two paragraphs how continuous batching changes latency and throughput "
    "trade-offs for autoregressive decoding."
)
MAX_NEW_TOKENS = 256
RUNS = 10


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _token_count(tokenizer: genai.Tokenizer, prompt: str) -> int:
    encoded = tokenizer.encode(prompt).input_ids
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    if encoded and hasattr(encoded[0], "__len__"):
        return len(encoded[0])
    return len(encoded)


def _make_generation_config() -> genai.GenerationConfig:
    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    cfg.do_sample = False
    return cfg


def run_vlm_once() -> dict[str, float]:
    pipe = genai.VLMPipeline(MODEL_DIR, DEVICE)
    cfg = _make_generation_config()
    input_tokens = _token_count(pipe.get_tokenizer(), PROMPT)

    t0 = time.perf_counter()
    result = pipe.generate(prompt=PROMPT, generation_config=cfg)
    t1 = time.perf_counter()

    metrics = result.perf_metrics
    ttft_s = metrics.get_ttft().mean / 1000.0
    decode_s = metrics.get_generate_duration().mean / 1000.0
    output_tokens = int(metrics.get_num_generated_tokens())

    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "ttft_s": ttft_s,
        "prefill_tps": _safe_div(float(input_tokens), ttft_s),
        "decode_s": decode_s,
        "decode_tps": _safe_div(float(output_tokens), decode_s),
        "total_s": t1 - t0,
    }


def run_cb_once() -> dict[str, float]:
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 2048
    scheduler.max_num_seqs = 1
    scheduler.cache_size = 4
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = True

    pipe = genai.ContinuousBatchingPipeline(
        MODEL_DIR,
        device=DEVICE,
        scheduler_config=scheduler,
    )

    cfg = _make_generation_config()
    input_tokens = _token_count(pipe.get_tokenizer(), PROMPT)

    t0 = time.perf_counter()
    handle = pipe.add_request(1, PROMPT, cfg)

    output_tokens = 0
    first_token_ts = None

    while True:
        pipe.step()

        if handle.can_read():
            for output in handle.read().values():
                new_tokens = len(output.generated_ids)
                if new_tokens > 0 and first_token_ts is None:
                    first_token_ts = time.perf_counter()
                output_tokens += new_tokens

        status = handle.get_status()
        if status in (
            genai.GenerationStatus.FINISHED,
            genai.GenerationStatus.IGNORED,
            genai.GenerationStatus.CANCEL,
            genai.GenerationStatus.STOP,
        ):
            break

    t1 = time.perf_counter()

    if first_token_ts is None:
        first_token_ts = t1

    ttft_s = first_token_ts - t0
    decode_s = max(t1 - first_token_ts, 0.0)

    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "ttft_s": ttft_s,
        "prefill_tps": _safe_div(float(input_tokens), ttft_s),
        "decode_s": decode_s,
        "decode_tps": _safe_div(float(output_tokens), decode_s),
        "total_s": t1 - t0,
    }


def average_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    keys = stats_list[0].keys()
    return {key: sum(stats[key] for stats in stats_list) / len(stats_list) for key in keys}


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(name)
    print(f"  input_tokens: {int(stats['input_tokens'])}")
    print(f"  output_tokens: {int(stats['output_tokens'])}")
    print(f"  ttft_s: {stats['ttft_s']:.6f}")
    print(f"  prefill_tps: {stats['prefill_tps']:.2f}")
    print(f"  decode_s: {stats['decode_s']:.6f}")
    print(f"  decode_tps: {stats['decode_tps']:.2f}")
    print(f"  total_s: {stats['total_s']:.6f}")


def main() -> None:
    vlm_runs: list[dict[str, float]] = []
    cb_runs: list[dict[str, float]] = []

    for i in range(1, RUNS + 1):
        print(f"Run {i}/{RUNS}: VLMPipeline")
        vlm_runs.append(run_vlm_once())

    for i in range(1, RUNS + 1):
        print(f"Run {i}/{RUNS}: ContinuousBatchingPipeline")
        cb_runs.append(run_cb_once())

    vlm_avg = average_stats(vlm_runs)
    cb_avg = average_stats(cb_runs)

    print_stats(f"VLMPipeline (avg of {RUNS})", vlm_avg)
    print_stats(f"ContinuousBatchingPipeline (avg of {RUNS})", cb_avg)

    print("Ratios (CB / VLM)")
    print(f"  prefill_tps_ratio: {_safe_div(cb_avg['prefill_tps'], vlm_avg['prefill_tps']):.3f}x")
    print(f"  decode_tps_ratio: {_safe_div(cb_avg['decode_tps'], vlm_avg['decode_tps']):.3f}x")


main()
