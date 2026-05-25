import json
from pathlib import Path
from typing import Any, Iterable


ARCHITECTURE_VISION_TOKENS = {
    "Gemma4ForConditionalGeneration": "<|image><|image|><image|>",
    "Gemma3ForConditionalGeneration": "<start_of_image>",
    "Qwen3_5ForConditionalGeneration": "<|vision_start|><|image_pad|><|vision_end|>",
    "Qwen3VLForConditionalGeneration": "<|vision_start|><|image_pad|><|vision_end|>",
    "Qwen2_5_VLForConditionalGeneration": "<|vision_start|><|image_pad|><|vision_end|>",
    "Qwen2VLForConditionalGeneration": "<|vision_start|><|image_pad|><|vision_end|>",
    "InternVLChatModel": "<image>",
    "InternVLForConditionalGeneration": "<image>",
    "LlavaForConditionalGeneration": "<image>",
    "LlavaNextForConditionalGeneration": "<image>",
    "MiniCPMV": "(<image>./</image>)",
    "MiniCPMVForCausalLM": "(<image>./</image>)",
    "Phi3VForCausalLM": "<|image_{i}|>",
    "Phi3VisionForCausalLM": "<|image_{i}|>",
    "Phi4MMForCausalLM": "<|image_{i}|>",
}


def _architecture_values(config: Any) -> Iterable[str]:
    if not isinstance(config, dict):
        return ()

    architectures = config.get("architectures")
    if isinstance(architectures, str):
        return (architectures,)
    if isinstance(architectures, list):
        return tuple(item for item in architectures if isinstance(item, str))
    return ()


def _token_from_architectures(architectures: Iterable[str]) -> str | None:
    for architecture in architectures:
        token = ARCHITECTURE_VISION_TOKENS.get(architecture)
        if token is not None:
            return token
    return None


def _token_from_raw_config(config_text: str) -> str | None:
    for architecture, token in ARCHITECTURE_VISION_TOKENS.items():
        if architecture in config_text:
            return token
    return None


def _drop_model_type(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_model_type(item)
            for key, item in value.items()
            if key != "model_type"
        }
    if isinstance(value, list):
        return [_drop_model_type(item) for item in value]
    return value


def supported_architectures() -> list[str]:
    return list(ARCHITECTURE_VISION_TOKENS.keys())


def resolve_vlm_vision_token(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"

    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Could not read VLM config.json at {config_path}. Supported architectures: "
            f"{', '.join(supported_architectures())}"
        ) from exc

    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        config = None

    architecture_token = _token_from_architectures(_architecture_values(config))
    if architecture_token is not None:
        return architecture_token

    scan_text = config_text
    if config is not None:
        scan_text = json.dumps(_drop_model_type(config))

    scan_token = _token_from_raw_config(scan_text)
    if scan_token is not None:
        return scan_token

    raise ValueError(
        f"Could not resolve VLM vision token from {config_path}. Supported architectures: "
        f"{', '.join(supported_architectures())}"
    )
