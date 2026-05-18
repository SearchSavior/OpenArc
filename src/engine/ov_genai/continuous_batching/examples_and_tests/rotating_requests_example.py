from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass, field

import openvino_genai as genai


@dataclass
class IncomingRequest:
    """Represents a mock external request waiting to be admitted into the pipeline."""

    request_id: int
    arrival_step: int
    prompt: str
    use_input_ids: bool
    target_input_tokens: int
    actual_input_tokens: int

@dataclass
class ActiveRequest:
    """Tracks an in-flight generation request and the tokens produced so far."""

    request_id: int
    prompt: str
    submitted_step: int
    via_input_ids: bool
    handle: genai.GenerationHandle
    generated_ids: list[int] = field(default_factory=list)
    finished_step: int | None = None
    finish_reason: genai.GenerationFinishReason | None = None


def build_pipeline(model_dir: str, device: str) -> genai.ContinuousBatchingPipeline:
    """Create a continuous batching pipeline configured for the demo workload."""

    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 2048
    scheduler.max_num_seqs = 8
    scheduler.cache_size = 4
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = True

    return genai.ContinuousBatchingPipeline(
        model_dir,
        device=device,
        scheduler_config=scheduler,
    )


def build_generation_config(max_new_tokens: int) -> genai.GenerationConfig:
    """Return a deterministic generation config for reproducible behavior."""

    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    return cfg


def sample_poisson(rng: random.Random, lam: float) -> int:
    """Sample an integer from a Poisson distribution using Knuth's algorithm."""

    if lam <= 0.0:
        return 0

    threshold = pow(2.718281828459045, -lam)
    k = 0
    product = 1.0
    while product > threshold:
        k += 1
        product *= rng.random()
    return k - 1


def count_tokens(tokenizer: genai.Tokenizer, prompt: str) -> int:
    """Count input tokens for a prompt with the pipeline tokenizer."""

    return int(tokenizer.encode(prompt).input_ids.shape[-1])


def build_prompt_for_target_tokens(
    tokenizer: genai.Tokenizer,
    request_id: int,
    target_tokens: int,
) -> tuple[str, int]:
    """Build a prompt that reaches roughly the requested input token budget."""

    base_prompt = (
        f"Request {request_id}: Give one concise sentence about continuous batching fairness "
        f"and throughput trade-offs. Context: "
    )
    current_prompt = base_prompt
    current_tokens = count_tokens(tokenizer, current_prompt)
    if current_tokens >= target_tokens:
        return current_prompt, current_tokens

    filler_words = (
        "latency throughput fairness queueing prefill decode batching scheduler "
        "cache memory arbitration tail performance admission policy service-level "
    ).split()
    idx = 0
    while current_tokens < target_tokens:
        current_prompt += filler_words[idx % len(filler_words)] + " "
        current_tokens = count_tokens(tokenizer, current_prompt)
        idx += 1

    return current_prompt, current_tokens


def make_mock_incoming_requests(
    tokenizer: genai.Tokenizer,
    total_requests: int,
    seed: int,
    input_tokens_min: int,
    input_tokens_max: int,
    poisson_lambda: float,
) -> deque[IncomingRequest]:
    """Create staggered incoming traffic with bounded Poisson burstiness in input size."""

    rng = random.Random(seed)
    arrivals: list[IncomingRequest] = []
    step_cursor = 0

    for request_id in range(1, total_requests + 1):
        step_cursor += rng.randint(0, 2)
        use_input_ids = request_id % 2 == 0
        poisson_draw = sample_poisson(rng, poisson_lambda)
        target_input_tokens = max(input_tokens_min, min(input_tokens_max, poisson_draw))
        prompt, actual_input_tokens = build_prompt_for_target_tokens(
            tokenizer=tokenizer,
            request_id=request_id,
            target_tokens=target_input_tokens,
        )
        arrivals.append(
            IncomingRequest(
                request_id=request_id,
                arrival_step=step_cursor,
                prompt=prompt,
                use_input_ids=use_input_ids,
                target_input_tokens=target_input_tokens,
                actual_input_tokens=actual_input_tokens,
            )
        )

    return deque(arrivals)


def submit_request(
    pipeline: genai.ContinuousBatchingPipeline,
    cfg: genai.GenerationConfig,
    incoming: IncomingRequest,
    current_step: int,
) -> ActiveRequest:
    """Submit one incoming request through either prompt or input_ids add_request overload."""

    if incoming.use_input_ids:
        tokenized = pipeline.get_tokenizer().encode(incoming.prompt)
        handle = pipeline.add_request(incoming.request_id, tokenized.input_ids, cfg)
    else:
        handle = pipeline.add_request(incoming.request_id, incoming.prompt, cfg)

    return ActiveRequest(
        request_id=incoming.request_id,
        prompt=incoming.prompt,
        submitted_step=current_step,
        via_input_ids=incoming.use_input_ids,
        handle=handle,
    )


def poll_active_requests(
    active: dict[int, ActiveRequest],
    current_step: int,
) -> list[ActiveRequest]:
    """Read newly generated chunks from each handle and return requests that reached terminal state."""

    finished: list[ActiveRequest] = []

    for req in active.values():
        if req.handle.can_read():
            for output in req.handle.read().values():
                req.generated_ids.extend(output.generated_ids)
                if output.finish_reason != genai.GenerationFinishReason.NONE:
                    req.finish_reason = output.finish_reason

        status = req.handle.get_status()
        if status in (
            genai.GenerationStatus.FINISHED,
            genai.GenerationStatus.IGNORED,
            genai.GenerationStatus.CANCEL,
            genai.GenerationStatus.STOP,
        ):
            req.finished_step = current_step
            if req.finish_reason is None:
                req.finish_reason = genai.GenerationFinishReason.NONE
            finished.append(req)

    return finished


def run_rotating_demo(
    pipeline: genai.ContinuousBatchingPipeline,
    cfg: genai.GenerationConfig,
    incoming_queue: deque[IncomingRequest],
    max_inflight: int,
) -> list[ActiveRequest]:
    """Simulate a rotating active set by admitting arrivals and retiring completed requests."""

    active: dict[int, ActiveRequest] = {}
    completed: list[ActiveRequest] = []
    step_idx = 0

    while incoming_queue or active:
        while incoming_queue and incoming_queue[0].arrival_step <= step_idx and len(active) < max_inflight:
            incoming = incoming_queue.popleft()
            req = submit_request(pipeline, cfg, incoming, current_step=step_idx)
            active[req.request_id] = req
            path = "input_ids" if req.via_input_ids else "prompt"
            print(
                f"[step {step_idx:04d}] admitted request={req.request_id:03d} "
                f"via={path} input_tokens={incoming.actual_input_tokens:03d} "
                f"target={incoming.target_input_tokens:03d} active={len(active):02d}/{max_inflight}"
            )

        if active:
            pipeline.step()

        finished_now = poll_active_requests(active, current_step=step_idx)
        for req in finished_now:
            active.pop(req.request_id, None)
            completed.append(req)
            latency_steps = (req.finished_step - req.submitted_step) if req.finished_step is not None else -1
            print(
                f"[step {step_idx:04d}] finished request={req.request_id:03d} "
                f"tokens={len(req.generated_ids):03d} latency_steps={latency_steps:03d} "
                f"reason={req.finish_reason.name if req.finish_reason else 'UNKNOWN'} "
                f"active={len(active):02d}/{max_inflight}"
            )

        step_idx += 1

    return completed


def print_summary(pipeline: genai.ContinuousBatchingPipeline, completed: list[ActiveRequest]) -> None:
    """Print final per-request summaries and decode a short text sample for inspection."""

    tokenizer = pipeline.get_tokenizer()

    print("\n=== Completed Requests ===")
    for req in sorted(completed, key=lambda item: item.request_id):
        sample = tokenizer.decode(req.generated_ids[:64]).strip() if req.generated_ids else ""
        path = "input_ids" if req.via_input_ids else "prompt"
        print(
            f"request={req.request_id:03d} via={path} out_tokens={len(req.generated_ids):03d} "
            f"finish_reason={req.finish_reason.name if req.finish_reason else 'UNKNOWN'}"
        )
        print(f"sample: {sample or '<no decoded text>'}")


def main() -> None:
    """CLI entrypoint for the rotating-request tracking example."""

    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate tracking a rotating set of mock incoming requests with "
            "ContinuousBatchingPipeline + GenerationHandle."
        )
    )
    parser.add_argument("model_dir", help="Path to OpenVINO model directory")
    parser.add_argument("--device", default="GPU.0", help="OpenVINO device")
    parser.add_argument("--total-requests", type=int, default=12, help="Number of mock requests")
    parser.add_argument("--max-inflight", type=int, default=4, help="Active request capacity")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Generation length per request")
    parser.add_argument("--seed", type=int, default=7, help="Seed for mock request arrivals")
    parser.add_argument(
        "--input-tokens-min",
        type=int,
        default=256,
        help="Lower bound for sampled input token size per request",
    )
    parser.add_argument(
        "--input-tokens-max",
        type=int,
        default=2048,
        help="Upper bound for sampled input token size per request",
    )
    parser.add_argument(
        "--poisson-lambda",
        type=float,
        default=96.0,
        help="Poisson lambda used to sample bursty input token sizes",
    )
    args = parser.parse_args()

    if args.input_tokens_min <= 0:
        raise ValueError("--input-tokens-min must be > 0")
    if args.input_tokens_max < args.input_tokens_min:
        raise ValueError("--input-tokens-max must be >= --input-tokens-min")
    if args.poisson_lambda <= 0.0:
        raise ValueError("--poisson-lambda must be > 0")

    pipeline = build_pipeline(args.model_dir, args.device)
    cfg = build_generation_config(args.max_new_tokens)
    incoming = make_mock_incoming_requests(
        tokenizer=pipeline.get_tokenizer(),
        total_requests=args.total_requests,
        seed=args.seed,
        input_tokens_min=args.input_tokens_min,
        input_tokens_max=args.input_tokens_max,
        poisson_lambda=args.poisson_lambda,
    )

    completed = run_rotating_demo(
        pipeline=pipeline,
        cfg=cfg,
        incoming_queue=incoming,
        max_inflight=args.max_inflight,
    )
    print_summary(pipeline, completed)


if __name__ == "__main__":
    main()
