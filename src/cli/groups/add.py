"""
Add command - Add a model configuration to the config file.
"""
import json

import click
from pydantic import ValidationError

from src.server.schemas.modeling.config_blocks import validate_block
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import SchedulerConfigSchema

from ..main import cli, console
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
@click.option("--scheduler-config", "-sc",
    default=None,
    help='OpenVINO runtime scheduler configuration as JSON string (e.g., \'{"use_sparse_attention": true}\').')
@click.option("--sampler-config", "--smc",
    default=None,
    help='Default sampler settings for llm/vlm models as JSON string (e.g., \'{"temperature": 0.7, "top_k": 40}\'). Overridden per request.')
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
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, scheduler_config, sampler_config, cache_dir, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold, tool_call_parser):
    """- Add a model configuration to the config file."""

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
    parsed_scheduler_config = {}
    if scheduler_config:
        # Let the model validate the JSON itself. If it validates, assume we can safely load the JSON.
        try:
            parsed_scheduler_config = json.loads(scheduler_config)
            if not isinstance(parsed_scheduler_config, dict):
                console.print(f"[red]Error: scheduler_config must be a JSON object (dictionary), got {type(scheduler_config).__name__}[/red]")
                console.print('[yellow]Example format: \'{"max_num_batched_tokens": 256, "enable_prefix_caching": true}\'[/yellow]')
            SchedulerConfigSchema.model_validate_json(scheduler_config)
        except ValidationError as e:
                console.print("[red]Error: Failed validating scheduler_config:[/red]")
                console.print('[yellow]Example format: \'{"max_num_batched_tokens": 256, "enable_prefix_caching": true}\'[/yellow]')
                console.print('')
                console.print('[yellow]Error:[/yellow]')
                console.print(e)
                ctx.exit(1)

    # Validate against the contract that backs the block, so an invalid default
    # is caught here rather than at model load time.
    parsed_sampler_config = {}
    if sampler_config:
        try:
            parsed_sampler_config = json.loads(sampler_config)
            if not isinstance(parsed_sampler_config, dict):
                console.print(f"[red]Error: sampler_config must be a JSON object (dictionary), got {type(parsed_sampler_config).__name__}[/red]")
                console.print('[yellow]Example format: \'{"temperature": 0.7, "top_k": 40}\'[/yellow]')
                ctx.exit(1)
            validate_block("sampler_config", parsed_sampler_config, model_type)
        except (json.JSONDecodeError, ValidationError) as e:
            console.print("[red]Error: Failed validating sampler_config:[/red]")
            console.print('[yellow]Example format: \'{"temperature": 0.7, "top_k": 40}\'[/yellow]')
            console.print('')
            console.print('[yellow]Error:[/yellow]')
            console.print(e)
            ctx.exit(1)

    # Model-level request defaults are stored as siblings of load_config.
    entry = {
        "load_config": {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "engine": engine,
            "device": device,
            "runtime_config": parsed_runtime_config,
            "scheduler_config": parsed_scheduler_config,
        }
    }
    if parsed_sampler_config:
        entry["sampler_config"] = parsed_sampler_config

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
