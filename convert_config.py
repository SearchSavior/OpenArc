"""Convert a legacy openarc_config.json to the config.yaml format.

The legacy JSON stored each model as a flat mapping with runtime_config and
scheduler_config nested inside it. The YAML format nests load-time fields
under `load_config` and keeps `runtime_config` / `scheduler_config` as
siblings, matching what `openarc add` writes.

Usage:
    python scripts/convert_config.py [input.json] [-o config.yaml] [--force]

Dropped from the legacy format: `mcp`, `vlm_type`, `created_by`, `version`,
and empty runtime_config/scheduler_config dicts.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

VALID_MODEL_TYPES = {
    "llm", "vlm", "whisper", "qwen3_asr", "kokoro",
    "qwen3_tts_custom_voice", "qwen3_tts_voice_design",
    "qwen3_tts_voice_clone", "emb", "rerank",
}
VALID_ENGINES = {"ovgenai", "openvino", "optimum"}

LOAD_FIELDS = [
    "model_path",
    "model_type",
    "engine",
    "device",
    "tool_call_parser",
    "cache_dir",
    "draft_model_path",
    "draft_device",
    "num_assistant_tokens",
    "assistant_confidence_threshold",
]

# Handled elsewhere or intentionally dropped; anything else warns.
KNOWN_NON_LOAD_KEYS = {
    "model_name",          # rewritten from the mapping key
    "runtime_config",      # becomes a sibling block
    "scheduler_config",    # becomes a sibling block
    "vlm_type",            # legacy, dropped
}


def _coerce_scalars(block):
    """Coerce numeric strings ('1' -> 1) so scheduler values keep their types."""
    coerced = {}
    for key, value in block.items():
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        coerced[key] = value
    return coerced


def convert_model(name, old, warnings):
    for key in old:
        if key not in LOAD_FIELDS and key not in KNOWN_NON_LOAD_KEYS:
            warnings.append(f"{name}: unknown key '{key}' passed through to load_config")

    model_type = old.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        warnings.append(
            f"{name}: skipped - unknown model_type '{model_type}' "
            f"(valid: {', '.join(sorted(VALID_MODEL_TYPES))})"
        )
        return None

    engine = old.get("engine")
    if engine not in VALID_ENGINES:
        warnings.append(
            f"{name}: skipped - unknown engine '{engine}' "
            f"(valid: {', '.join(sorted(VALID_ENGINES))})"
        )
        return None

    load_config = {"model_name": name}
    for field in LOAD_FIELDS:
        if field in old and old[field] is not None:
            load_config[field] = old[field]

    entry = {"load_config": load_config}

    runtime_config = old.get("runtime_config") or {}
    if runtime_config:
        entry["runtime_config"] = runtime_config

    scheduler_config = old.get("scheduler_config") or {}
    if scheduler_config:
        entry["scheduler_config"] = _coerce_scalars(scheduler_config)

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy openarc_config.json to config.yaml"
    )
    parser.add_argument(
        "input", nargs="?", default="openarc_config.json",
        help="Path to the legacy JSON config (default: openarc_config.json)",
    )
    parser.add_argument(
        "-o", "--output", default="config.yaml",
        help="Path to write the YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it exists",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        parser.error(f"{output_path} already exists; pass --force to overwrite")

    with open(input_path, "r", encoding="utf-8") as f:
        old = json.load(f)

    warnings = []
    new = {}

    server = old.get("server")
    if isinstance(server, dict) and server:
        new["server"] = server

    models = {}
    skipped = 0
    for name, entry in old.get("models", {}).items():
        converted = convert_model(name, entry, warnings)
        if converted is None:
            skipped += 1
            continue
        models[name] = converted
    new["models"] = models

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {len(models)} model(s) to {output_path}")
    if skipped:
        print(f"Skipped {skipped} model(s)")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
