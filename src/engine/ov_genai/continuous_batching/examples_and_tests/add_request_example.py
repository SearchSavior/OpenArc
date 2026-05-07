from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import openvino as ov
import openvino_genai as genai


@dataclass
class RequestSpec:
    """Describes a submitted request so outputs can be labeled consistently."""

    request_id: int
    label: str


def build_pipeline(model_dir: str, device: str) -> genai.ContinuousBatchingPipeline:
    """Create a continuous batching pipeline with a small, example-friendly scheduler."""

    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 2048
    scheduler.max_num_seqs = 16
    scheduler.cache_size = 4
    scheduler.dynamic_split_fuse = True
    scheduler.enable_prefix_caching = True

    return genai.ContinuousBatchingPipeline(
        model_dir,
        device=device,
        scheduler_config=scheduler,
    )


def build_generation_config(max_new_tokens: int = 48) -> genai.GenerationConfig:
    """Create a deterministic generation config used by all demo requests."""

    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    return cfg


def drain_requests(
    pipeline: genai.ContinuousBatchingPipeline,
    handles: dict[int, genai.GenerationHandle],
) -> dict[int, list[int]]:
    """Run pipeline steps until completion and collect generated token IDs per request."""

    generated_ids: dict[int, list[int]] = {req_id: [] for req_id in handles}

    while pipeline.has_non_finished_requests():
        pipeline.step()
        for req_id, handle in handles.items():
            if not handle.can_read():
                continue
            for output in handle.read().values():
                generated_ids[req_id].extend(output.generated_ids)

    return generated_ids


def add_text_request(
    pipeline: genai.ContinuousBatchingPipeline,
    request_id: int,
    prompt: str,
    cfg: genai.GenerationConfig,
) -> genai.GenerationHandle:
    """Submit a request through the prompt-string add_request overload."""

    # add_request(request_id, prompt: str, generation_config)
    return pipeline.add_request(request_id, prompt, cfg)


def add_input_ids_request(
    pipeline: genai.ContinuousBatchingPipeline,
    request_id: int,
    prompt: str,
    cfg: genai.GenerationConfig,
) -> genai.GenerationHandle:
    """Tokenize text first, then submit through the input_ids add_request overload."""

    # add_request(request_id, input_ids: ov.Tensor, generation_config)
    tokenizer = pipeline.get_tokenizer()
    tokenized = tokenizer.encode(prompt)
    return pipeline.add_request(request_id, tokenized.input_ids, cfg)


def add_multimodal_request(
    pipeline: genai.ContinuousBatchingPipeline,
    request_id: int,
    prompt: str,
    image_tensors: Sequence[ov.Tensor],
    cfg: genai.GenerationConfig,
) -> genai.GenerationHandle:
    """Submit a prompt+images request for VLM-capable continuous batching pipelines."""

    # add_request(request_id, prompt: str, images: Sequence[ov.Tensor], generation_config)
    # This overload is for ContinuousBatchingPipeline with a VLM-capable model.
    return pipeline.add_request(request_id, prompt, list(image_tensors), cfg)


def make_demo_image() -> ov.Tensor:
    """Create a tiny synthetic image tensor used to exercise the multimodal overload."""

    # Small synthetic RGB image (HWC, uint8) for API demonstration.
    return ov.Tensor(ov.Type.u8, ov.Shape([32, 32, 3]))


def print_decoded_outputs(
    pipeline: genai.ContinuousBatchingPipeline,
    requests: list[RequestSpec],
    generated_by_request: dict[int, list[int]],
) -> None:
    """Decode generated token IDs and print labeled text for each submitted request."""

    tokenizer = pipeline.get_tokenizer()

    for req in requests:
        token_ids = generated_by_request[req.request_id]
        decoded = tokenizer.decode(token_ids) if token_ids else ""
        print(f"[{req.label}] request_id={req.request_id}")
        print(decoded.strip() or "<no decoded text>")
        print()


def main() -> None:
    """Parse CLI args, submit demo requests, and print decoded generation outputs."""

    parser = argparse.ArgumentParser(
        description="Demonstrate ContinuousBatchingPipeline.add_request overloads."
    )
    parser.add_argument("model_dir", help="Path to OpenVINO model directory")
    parser.add_argument("--device", default="CPU", help="OpenVINO device")
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Also submit the multimodal add_request overload (requires a VLM model)",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args.model_dir, args.device)
    cfg = build_generation_config()

    requests: list[RequestSpec] = [
        RequestSpec(1, "text prompt overload"),
        RequestSpec(2, "input_ids overload"),
    ]

    handles: dict[int, genai.GenerationHandle] = {}
    handles[1] = add_text_request(
        pipeline,
        request_id=1,
        prompt="Write one short sentence about continuous batching.",
        cfg=cfg,
    )
    handles[2] = add_input_ids_request(
        pipeline,
        request_id=2,
        prompt="Write one short sentence about tokenization.",
        cfg=cfg,
    )

    if args.include_multimodal:
        requests.append(RequestSpec(3, "multimodal prompt+images overload"))
        demo_image = make_demo_image()
        handles[3] = add_multimodal_request(
            pipeline,
            request_id=3,
            prompt="Describe the image in one short sentence.",
            image_tensors=[demo_image],
            cfg=cfg,
        )

    generated_by_request = drain_requests(pipeline, handles)
    print_decoded_outputs(pipeline, requests, generated_by_request)


if __name__ == "__main__":
    main()
