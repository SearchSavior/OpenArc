"""
Add command - Add a model configuration to the config file.
"""
import json

import click

# rich_click's package namespace falls through to click for unknown names, so
# the globals module is imported directly. Mutating its dict in place is what
# RichHelpConfiguration.load_from_globals() reads.
from rich_click.rich_click import OPTION_GROUPS as RICH_CLICK_OPTION_GROUPS

from ..main import cli, console
from ..modules.config_options import (
    ConfigOptionError,
    config_options,
    option_groups,
    resolve_config_values,
)
from ..utils import validate_model_path


@cli.command()
@click.option('--model-name', '--mn',
    required=True,
    help='Public facing name of the model.')
@click.option('--model-path', '--m',
    required=True,
    help='Path to OpenVINO IR converted model.')
@click.option('--engine', '--en',
    type=click.Choice(['ovgenai', 'openvino', 'optimum']),
    required=True,
    help='Engine used to load the model (ovgenai, openvino, optimum)')
@click.option('--model-type', '--mt',
    type=click.Choice([
        'llm', 'vlm', 'whisper', 'qwen3_asr', 'kokoro',
        'qwen3_tts_custom_voice', 'qwen3_tts_voice_design', 'qwen3_tts_voice_clone',
        'emb', 'rerank',
    ]),
    required=True,
    help='Model type (llm, vlm, whisper, qwen3_asr, kokoro, qwen3_tts_custom_voice, qwen3_tts_voice_design, qwen3_tts_voice_clone, emb, rerank)')
@click.option('--device', '--d',
    required=True,
    help='Device(s) to load the model on.')
@click.option("--runtime-config", "--rtc",
    default=None,
    help='OpenVINO runtime configuration as JSON string (e.g., \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\').')
@click.option('--cache-dir', '--cd',
    required=False,
    default=None,
    help='Directory for the OpenVINO model cache. Caching compiled model blobs here speeds up subsequent loads of this model. Relative paths are resolved against the config file, like --model-path.')
@click.option('--draft-model-path', '--dmp',
    required=False,
    default=None,
    help='Path to draft model for speculative decoding.')
@click.option('--draft-device', '--dd',
    required=False,
    default=None,
    help='Draft model device.')
@click.option('--num-assistant-tokens', '--nat',
    required=False,
    default=None,
    type=int,
    help='Number of tokens draft model generates per step.')
@click.option('--assistant-confidence-threshold', '--act',
    required=False,
    default=None,
    type=float,
    help='Confidence threshold for accepting draft tokens.')
@click.option('--tool-call-parser',
    type=click.Choice(['qwen35', 'hermes', 'gemma4', 'museglimmer']),
    required=False,
    default=None,
    help='Tool-call output format for this model (qwen35 XML, hermes JSON, gemma4 call syntax, or museglimmer Harmony atem). llm/vlm only; required for tool calling.')
@config_options
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, cache_dir, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold, tool_call_parser, **config_values):
    """- Add a model configuration to the config file.

    \b
    Model defaults (--temperature, --max-tokens, --max-num-seqs, ...) are
    generated from the pydantic contracts in src/server/schemas/modeling, so
    they never drift from the request schema. Each help panel below is one
    config.yaml key. A flag is written to the key that backs it for the chosen
    --model-type, and a flag that does not apply to that model type is
    rejected. Only flags you actually pass are written, so everything else
    keeps using the contract's own default.
    """

    # Validate model path
    if not validate_model_path(model_path):
        console.print(f"[red]Model file check failed! {model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
        ctx.exit(1)

    # Parse runtime_config if provided
    parsed_runtime_config = {}
    if runtime_config:
        try:
            parsed_runtime_config = json.loads(runtime_config)
            if not isinstance(parsed_runtime_config, dict):
                console.print(f"[red]Error: runtime_config must be a JSON object (dictionary), got {type(parsed_runtime_config).__name__}[/red]")
                console.print('[yellow]Example format: \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\'[/yellow]')
                ctx.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing runtime_config JSON:[/red] {e}")
            console.print('[yellow]Example format: \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\'[/yellow]')
            ctx.exit(1)
    # Route the contract-flagged values into their blocks. Validation happens
    # against the contracts themselves, so a bad value is caught here rather
    # than at model load time.
    try:
        scheduler_config, blocks = resolve_config_values(model_type, config_values)
    except ConfigOptionError as e:
        console.print("[red]Error: invalid model configuration options:[/red]")
        console.print(e)
        ctx.exit(1)

    entry = {
        "load_config": {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "engine": engine,
            "device": device,
            "runtime_config": parsed_runtime_config,
        }
    }
    if scheduler_config:
        entry["load_config"]["scheduler_config"] = scheduler_config

    # Model-level request defaults are stored as siblings of load_config.
    for block_name, payload in blocks.items():
        entry[block_name] = payload

    # Store the cache directory (resolved relative to the config file at load time)
    if cache_dir:
        entry["load_config"]["cache_dir"] = cache_dir

    # Add speculative decoding options if provided
    if draft_model_path:
        if not validate_model_path(draft_model_path):
            console.print(f"[red]Model file check failed! {draft_model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
            ctx.exit(1)
        entry["load_config"]["draft_model_path"] = draft_model_path
    if draft_device:
        entry["load_config"]["draft_device"] = draft_device
    if num_assistant_tokens is not None:
        entry["load_config"]["num_assistant_tokens"] = num_assistant_tokens
    if assistant_confidence_threshold is not None:
        entry["load_config"]["assistant_confidence_threshold"] = assistant_confidence_threshold
    if tool_call_parser:
        entry["load_config"]["tool_call_parser"] = tool_call_parser

    ctx.obj.server_config.save_model_entry(model_name, entry)
    console.print(f"[green]Model configuration saved:[/green] {model_name}")
    console.print(f"[dim]Use 'openarc load {model_name}' to load this model.[/dim]")


# Hand-declared load_config flags, plus click's own --help. Everything else in
# the help comes from the contracts via option_groups().
_LOAD_OPTIONS = [
    "help",
    "model_name",
    "model_path",
    "engine",
    "model_type",
    "device",
    "runtime_config",
    "cache_dir",
    "draft_model_path",
    "draft_device",
    "num_assistant_tokens",
    "assistant_confidence_threshold",
    "tool_call_parser",
]

# One help panel per config.yaml key, keyed by the command path rich_click
# matches against. This mutates rich_click's own groups, so there is no custom
# panel code to maintain.
RICH_CLICK_OPTION_GROUPS["*add"] = option_groups(_LOAD_OPTIONS)
