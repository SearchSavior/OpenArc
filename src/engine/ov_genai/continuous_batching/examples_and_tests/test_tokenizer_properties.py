from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openvino as ov
import openvino_genai as genai


DEFAULT_MODEL_DIR = (
    "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/"
    "Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov/"
)


@dataclass
class PropertyCandidate:
    name: str
    props: dict[str, Any]
    category: str = "valid"


@dataclass
class ProbeResult:
    name: str
    ok: bool
    elapsed_ms: float
    error: str | None


def _mk_candidates() -> list[PropertyCandidate]:
    """Build tokenizer property candidates with type-safe OV helpers."""
    candidates: list[PropertyCandidate] = []

    def add(name: str, kv: tuple[str, Any], category: str = "valid") -> None:
        k, v = kv
        candidates.append(PropertyCandidate(name=name, props={k: v}, category=category))

    add("PERF_COUNT=True", ov.properties.enable_profiling(True))
    add("INFERENCE_NUM_THREADS=4", ov.properties.inference_num_threads(4))
    add("NUM_STREAMS=1", ov.properties.num_streams(ov.properties.streams.Num(1)))
    add(
        "PERFORMANCE_HINT=LATENCY",
        ov.properties.hint.performance_mode(ov.properties.hint.PerformanceMode.LATENCY),
    )
    add(
        "EXECUTION_MODE_HINT=PERFORMANCE",
        ov.properties.hint.execution_mode(ov.properties.hint.ExecutionMode.PERFORMANCE),
    )
    add("PERFORMANCE_HINT_NUM_REQUESTS=1", ov.properties.hint.num_requests(1))
    add("ENABLE_CPU_PINNING=True", ov.properties.hint.enable_cpu_pinning(True))
    add(
        "ENABLE_CPU_RESERVATION=False",
        ov.properties.hint.enable_cpu_reservation(False),
    )
    add(
        "ENABLE_HYPER_THREADING=True",
        ov.properties.hint.enable_hyper_threading(True),
    )
    add(
        "SCHEDULING_CORE_TYPE=ANY_CORE",
        ov.properties.hint.scheduling_core_type(
            ov.properties.hint.SchedulingCoreType.ANY_CORE
        ),
    )
    add(
        "MODEL_DISTRIBUTION_POLICY={PIPELINE_PARALLEL}",
        ov.properties.hint.model_distribution_policy(
            {ov.properties.hint.ModelDistributionPolicy.PIPELINE_PARALLEL}
        ),
    )
    add(
        "CPU_DENORMALS_OPTIMIZATION=True",
        ov.properties.intel_cpu.denormals_optimization(True),
    )
    add(
        "TBB_PARTITIONER=AUTO",
        ov.properties.intel_cpu.tbb_partitioner(ov.properties.intel_cpu.TbbPartitioner.AUTO),
    )
    add(
        "DYNAMIC_QUANTIZATION_GROUP_SIZE=64",
        ov.properties.hint.dynamic_quantization_group_size(64),
    )
    add("LOG_LEVEL=WARNING", ov.properties.log.level(ov.properties.log.Level.WARNING))
    add("CACHE_MODE=OPTIMIZE_SPEED", ov.properties.cache_mode(ov.properties.CacheMode.OPTIMIZE_SPEED))
    add("CACHE_DIR=/tmp/ov_tokenizer_cache", ov.properties.cache_dir("/tmp/ov_tokenizer_cache"))

    # Unknown key probe.
    candidates.append(
        PropertyCandidate(
            name="INVALID_KEY=BLAH",
            props={"BLAH_NOT_A_REAL_OV_PROPERTY": 1},
            category="invalid_key",
        )
    )

    # Explicitly wrong value probes for known keys.
    candidates.append(
        PropertyCandidate(
            name="PERFORMANCE_HINT=NOT_A_MODE",
            props={"PERFORMANCE_HINT": "NOT_A_MODE"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="EXECUTION_MODE_HINT=NOPE",
            props={"EXECUTION_MODE_HINT": "NOPE"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="NUM_STREAMS=banana",
            props={"NUM_STREAMS": "banana"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="INFERENCE_NUM_THREADS=banana",
            props={"INFERENCE_NUM_THREADS": "banana"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="ENABLE_CPU_PINNING=banana",
            props={"ENABLE_CPU_PINNING": "banana"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL(str)",
            props={"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="TBB_PARTITIONER=INVALID",
            props={"TBB_PARTITIONER": "INVALID"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="LOG_LEVEL=LOUD",
            props={"LOG_LEVEL": "LOUD"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="CACHE_MODE=FASTEST",
            props={"CACHE_MODE": "FASTEST"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="DYNAMIC_QUANTIZATION_GROUP_SIZE=zero",
            props={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "zero"},
            category="invalid_value",
        )
    )
    candidates.append(
        PropertyCandidate(
            name="CPU_DENORMALS_OPTIMIZATION=banana",
            props={"CPU_DENORMALS_OPTIMIZATION": "banana"},
            category="invalid_value",
        )
    )
    return candidates


def _probe_tokenizer(model_dir: str, candidate: PropertyCandidate) -> ProbeResult:
    t0 = time.perf_counter()
    try:
        tok = genai.Tokenizer(model_dir, properties=candidate.props)
        encoded = tok.encode("Tokenizer property probe.")
        _ = int(encoded.input_ids.shape[-1])
        return ProbeResult(
            name=candidate.name,
            ok=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            name=candidate.name,
            ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _probe_cb_init(
    model_dir: str,
    device: str,
    candidate: PropertyCandidate,
) -> ProbeResult:
    t0 = time.perf_counter()
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 128
    scheduler.max_num_seqs = 1
    scheduler.cache_size = 1
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = False
    try:
        pipe = genai.ContinuousBatchingPipeline(
            model_dir,
            scheduler_config=scheduler,
            device=device,
            tokenizer_properties=candidate.props,
        )
        _ = pipe.get_tokenizer().encode("CB tokenizer_properties probe.")
        return ProbeResult(
            name=candidate.name,
            ok=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            name=candidate.name,
            ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_results(title: str, results: list[ProbeResult]) -> None:
    print(f"\n=== {title} ===")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        line = f"[{status}] {r.name} ({r.elapsed_ms:.1f} ms)"
        if r.error:
            line += f"\n  {r.error}"
        print(line)
    ok_count = sum(1 for r in results if r.ok)
    print(f"\nSummary: {ok_count}/{len(results)} passed")


def _print_validation_breakdown(
    title: str, candidates: list[PropertyCandidate], results: list[ProbeResult]
) -> None:
    by_name = {r.name: r for r in results}
    buckets: dict[str, tuple[int, int]] = {}
    for c in candidates:
        total, passed = buckets.get(c.category, (0, 0))
        total += 1
        if by_name.get(c.name) and by_name[c.name].ok:
            passed += 1
        buckets[c.category] = (total, passed)

    print(f"\n=== {title} Validation Breakdown ===")
    for category in sorted(buckets.keys()):
        total, passed = buckets[category]
        print(f"{category}: {passed}/{total} accepted")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe which OV properties are accepted in tokenizer_properties."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--cb-sample-count",
        type=int,
        default=4,
        help="How many candidates to probe through ContinuousBatchingPipeline init.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write raw results as JSON.",
    )
    args = parser.parse_args()

    candidates = _mk_candidates()
    tokenizer_results = [_probe_tokenizer(args.model_dir, c) for c in candidates]
    _print_results("Tokenizer(model_dir, properties=...)", tokenizer_results)
    _print_validation_breakdown("Tokenizer", candidates, tokenizer_results)

    cb_candidates = candidates[: max(0, args.cb_sample_count)]
    cb_results = [
        _probe_cb_init(args.model_dir, args.device, c)
        for c in cb_candidates
    ]
    _print_results(
        f"ContinuousBatchingPipeline(..., tokenizer_properties=...) on {args.device}",
        cb_results,
    )
    _print_validation_breakdown("CB tokenizer_properties", cb_candidates, cb_results)

    if args.output_json:
        payload = {
            "model_dir": args.model_dir,
            "device": args.device,
            "tokenizer_results": [r.__dict__ for r in tokenizer_results],
            "cb_results": [r.__dict__ for r in cb_results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
