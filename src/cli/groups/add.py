"""
Add command - Add a model configuration to the config file.
"""
import json

import click
from pydantic import ValidationError

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import SchedulerConfigSchema

from ..main import cli, console
from ..utils import validate_model_path


def _runtime_config_help_callback(ctx, _param, value):
    """Eager callback: ``--runtime-config --help`` prints the full, dynamically
    discovered key listing (from the installed openvino plugin) and exits, before
    the other required options of ``add`` are validated."""
    if value == "--help":
        from src.cli.modules.ov_config_docs import runtime_config_help

        console.print(runtime_config_help())
        ctx.exit(0)
    return value


def _scheduler_config_help_callback(ctx, _param, value):
    """Eager callback: ``--scheduler-config --help`` prints the full, dynamically
    discovered key listing (from the installed openvino_genai package) and exits,
    before the other required options of ``add`` are validated."""
    if value == "--help":
        from src.cli.modules.ov_config_docs import scheduler_config_help

        console.print(scheduler_config_help())
        ctx.exit(0)
    return value


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
    is_eager=True,
    callback=_runtime_config_help_callback,
    help=(
        'OpenVINO GPU runtime/compile configuration as a JSON string of plugin keys '
        '(e.g., \'{"OFFLOAD_RATIO": 0.05, "PERFORMANCE_HINT": "LATENCY"}\'). '
        'Run `openarc add --runtime-config --help` to list every available key with usage '
        '(discovered from the installed openvino plugin). Memory-relevant keys include '
        'OFFLOAD_RATIO, KV_CACHE_PRECISION, INFERENCE_PRECISION_HINT, NUM_STREAMS, '
        'PERFORMANCE_HINT.'))
@click.option("--scheduler-config", "-sc",
    default=None,
    is_eager=True,
    callback=_scheduler_config_help_callback,
    help=(
        'OpenVINO GenAI scheduler configuration as a JSON string of scheduler keys '
        '(e.g., \'{"cache_size": 12, "max_num_seqs": 1}\'). '
        'Run `openarc add --scheduler-config --help` to list every available key with usage '
        '(discovered from the installed openvino_genai package). Memory-relevant keys include '
        'cache_size, num_kv_blocks, max_num_seqs, max_num_batched_tokens.'))
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
@click.option('--context-window', '--cw',
    type=int,
    required=False,
    default=None,
    help='Context window (tokens) for this model. Becomes the compiled model\'s MAX CONTENT WINDOW (openvino.genai SchedulerConfig.max_num_batched_tokens) and is advertised in /v1/models. When omitted, the value is discovered from the model\'s config.json')
@click.option('--max-tokens',
    type=int,
    required=False,
    default=None,
    help='Model-level default max_tokens (max_new_tokens) applied when a request omits max_tokens. Bounds the output length for requests that do not specify one, avoiding the 16384 default and GPU OOM. An explicit client max_tokens always wins.')
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, scheduler_config, cache_dir, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold, tool_call_parser, context_window, max_tokens):
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

    # Legacy configs may still contain vlm_type, but new configs resolve VLM tokens from config.json.
    load_config = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type,
        "engine": engine,
        "device": device,
        "runtime_config": parsed_runtime_config,
        "scheduler_config": parsed_scheduler_config,
    }

    # Store the cache directory (resolved relative to the config file at load time)
    if cache_dir:
        load_config["cache_dir"] = cache_dir

    # Add speculative decoding options if provided
    if draft_model_path:
        if not validate_model_path(draft_model_path):
            console.print(f"[red]Model file check failed! {draft_model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
            ctx.exit(1)
        load_config["draft_model_path"] = draft_model_path
    if draft_device:
        load_config["draft_device"] = draft_device
    if num_assistant_tokens is not None:
        load_config["num_assistant_tokens"] = num_assistant_tokens
    if assistant_confidence_threshold is not None:
        load_config["assistant_confidence_threshold"] = assistant_confidence_threshold
    if tool_call_parser:
        load_config["tool_call_parser"] = tool_call_parser
    if context_window is not None:
        load_config["context_window"] = context_window
    if max_tokens is not None:
        load_config["max_tokens"] = max_tokens

    # A re-add that does not change the configuration keeps the stored
    # config_hash, so the next load does not needlessly recompile. Any change
    # drops the hash (the entry is replaced), which makes the next load
    # invalidate the compiled-model cache and recompile -- see
    # src.server.utils.config_hash. The comparison uses the raw file entry, so
    # the path strings compare exactly as typed.
    previous_entry = ctx.obj.server_config.load_config().get("models", {}).get(model_name)
    if isinstance(previous_entry, dict):
        same_config = (
            {k: v for k, v in previous_entry.items() if k != "config_hash"}
            == {k: v for k, v in load_config.items() if k != "config_hash"}
        )
        if same_config and previous_entry.get("config_hash"):
            load_config["config_hash"] = previous_entry["config_hash"]

    ctx.obj.server_config.save_model_config(model_name, load_config)
    console.print(f"[green]Model configuration saved:[/green] {model_name}")
    console.print(f"[dim]Use 'openarc load {model_name}' to load this model.[/dim]")
