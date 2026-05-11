from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openvino as ov
import openvino_genai as genai


DEFAULT_MODEL_DIR = "/mnt/Ironwolf-4TB/Models/OpenVINO/Gemma/gemma-3-4b-it-int4_asym-ov/"


@dataclass
class PropertyCandidate:
    name: str
    props: dict[str, Any]
    category: str = "valid"


@dataclass
class ProbeResult:
    device: str
    name: str
    field: str
    phase: str
    ok: bool
    elapsed_ms: float
    error: str | None


def _mk_candidates() -> list[PropertyCandidate]:
    candidates: list[PropertyCandidate] = []

    def add(name: str, kv: tuple[str, Any], category: str = "valid") -> None:
        k, v = kv
        candidates.append(PropertyCandidate(name=name, props={k: v}, category=category))

    # Valid-ish property probes (typed OV constructors).
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
        "MODEL_DISTRIBUTION_POLICY={PIPELINE_PARALLEL}",
        ov.properties.hint.model_distribution_policy(
            {ov.properties.hint.ModelDistributionPolicy.PIPELINE_PARALLEL}
        ),
    )
    add("LOG_LEVEL=WARNING", ov.properties.log.level(ov.properties.log.Level.WARNING))
    add("CACHE_MODE=OPTIMIZE_SPEED", ov.properties.cache_mode(ov.properties.CacheMode.OPTIMIZE_SPEED))
    add("CACHE_DIR=/tmp/ov_vision_encoder_cache", ov.properties.cache_dir("/tmp/ov_vision_encoder_cache"))
    add(
        "GPU_ENABLE_LOOP_UNROLLING=False",
        ov.properties.intel_gpu.enable_loop_unrolling(False),
    )
    add(
        "GPU_DISABLE_WINOGRAD_CONVOLUTION=True",
        ov.properties.intel_gpu.disable_winograd_convolution(True),
    )

    # Invalid key/value probes.
    candidates.append(
        PropertyCandidate(
            name="INVALID_KEY=BLAH",
            props={"BLAH_NOT_A_REAL_OV_PROPERTY": 1},
            category="invalid_key",
        )
    )
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
            name="ENABLE_CPU_PINNING=banana",
            props={"ENABLE_CPU_PINNING": "banana"},
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
            name="MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL(str)",
            props={"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"},
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
    return candidates


def _mk_scheduler() -> genai.SchedulerConfig:
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 64
    scheduler.max_num_seqs = 1
    scheduler.cache_size = 1
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = False
    return scheduler


def _probe_init_for_field(
    model_dir: str, device: str, candidate: PropertyCandidate, field: str
) -> ProbeResult:
    t0 = time.perf_counter()
    try:
        kwargs = {field: candidate.props}
        _ = genai.ContinuousBatchingPipeline(
            model_dir,
            scheduler_config=_mk_scheduler(),
            device=device,
            **kwargs,
        )
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="init",
            ok=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="init",
            ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _probe_runtime_vision_for_field(
    model_dir: str, device: str, candidate: PropertyCandidate, field: str
) -> ProbeResult:
    t0 = time.perf_counter()
    try:
        kwargs = {field: candidate.props}
        pipe = genai.ContinuousBatchingPipeline(
            model_dir,
            scheduler_config=_mk_scheduler(),
            device=device,
            **kwargs,
        )
        cfg = genai.GenerationConfig()
        cfg.max_new_tokens = 1
        cfg.do_sample = False
        img = ov.Tensor(ov.Type.u8, ov.Shape([32, 32, 3]))
        handle = pipe.add_request(1, "Describe image in one word.", [img], cfg)
        steps = 0
        while pipe.has_non_finished_requests() and steps < 8:
            pipe.step()
            if handle.can_read():
                _ = handle.read()
            steps += 1
        _ = handle.get_status()
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="runtime_vision",
            ok=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="runtime_vision",
            ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _probe_runtime_text_for_field(
    model_dir: str, device: str, candidate: PropertyCandidate, field: str
) -> ProbeResult:
    t0 = time.perf_counter()
    try:
        kwargs = {field: candidate.props}
        pipe = genai.ContinuousBatchingPipeline(
            model_dir,
            scheduler_config=_mk_scheduler(),
            device=device,
            **kwargs,
        )
        cfg = genai.GenerationConfig()
        cfg.max_new_tokens = 1
        cfg.do_sample = False
        handle = pipe.add_request(1, "One word.", cfg)
        steps = 0
        while pipe.has_non_finished_requests() and steps < 8:
            pipe.step()
            if handle.can_read():
                _ = handle.read()
            steps += 1
        _ = handle.get_status()
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="runtime_text",
            ok=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            device=device,
            name=candidate.name,
            field=field,
            phase="runtime_text",
            ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_results(title: str, results: list[ProbeResult]) -> None:
    print(f"\n=== {title} ===")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        line = (
            f"[{status}] {r.device} {r.field} {r.phase} "
            f"{r.name} ({r.elapsed_ms:.1f} ms)"
        )
        if r.error:
            line += f"\n  {r.error}"
        print(line)
    ok_count = sum(1 for r in results if r.ok)
    print(f"\nSummary: {ok_count}/{len(results)} passed")


def _print_validation_breakdown(
    title: str, candidates: list[PropertyCandidate], init_results: list[ProbeResult]
) -> None:
    by_name = {r.name: r for r in init_results}
    buckets: dict[str, tuple[int, int]] = {}
    for c in candidates:
        total, passed = buckets.get(c.category, (0, 0))
        total += 1
        if by_name.get(c.name) and by_name[c.name].ok:
            passed += 1
        buckets[c.category] = (total, passed)

    print(f"\n=== {title} Validation Breakdown (init) ===")
    for category in sorted(buckets.keys()):
        total, passed = buckets[category]
        print(f"{category}: {passed}/{total} accepted")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe which properties are validated in vision_encoder_properties "
            "and properties."
        )
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["CPU", "GPU"],
        help="Devices to probe, e.g. CPU GPU",
    )
    parser.add_argument(
        "--runtime-cases",
        type=int,
        default=3,
        help="Number of accepted init candidates to additionally probe via image request runtime.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    candidates = _mk_candidates()

    all_init_results: list[ProbeResult] = []
    all_runtime_results: list[ProbeResult] = []

    for device in args.devices:
        for field in ("vision_encoder_properties", "properties"):
            init_results = [
                _probe_init_for_field(args.model_dir, device, c, field)
                for c in candidates
            ]
            _print_results(f"{device} {field} init probes", init_results)
            _print_validation_breakdown(f"{device} {field}", candidates, init_results)
            all_init_results.extend(init_results)

            accepted_init = [
                c for c in candidates if any(r.name == c.name and r.ok for r in init_results)
            ]
            runtime_candidates = accepted_init[: max(0, args.runtime_cases)]
            if field == "vision_encoder_properties":
                runtime_results = [
                    _probe_runtime_vision_for_field(args.model_dir, device, c, field)
                    for c in runtime_candidates
                ]
            else:
                runtime_results = [
                    _probe_runtime_text_for_field(args.model_dir, device, c, field)
                    for c in runtime_candidates
                ]
            _print_results(f"{device} {field} runtime probes", runtime_results)
            all_runtime_results.extend(runtime_results)

    if args.output_json:
        payload = {
            "model_dir": args.model_dir,
            "devices": args.devices,
            "init_results": [r.__dict__ for r in all_init_results],
            "runtime_results": [r.__dict__ for r in all_runtime_results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
