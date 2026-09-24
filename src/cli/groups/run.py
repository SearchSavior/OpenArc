"""
Run command - Add a model config, start the server, and load the model in one step.
"""
import json
import os
import threading
import time

import click
import requests
from pydantic import ValidationError

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
@click.option("--host", type=str, default="0.0.0.0", show_default=True,
              help='Host to bind the server to')
@click.option("--port", type=int, default=8000, show_default=True,
              help='Port to bind the server to')
@click.option("--use-api-key", is_flag=True, default=False,
              help="Require OPENARC_API_KEY for all requests.")
@click.option("-v", "--verbose", count=True, default=0,
              help="Increase verbosity: -v warnings, -vv info + HTTP requests, -vvv debug, -vvvv debug incl. third-party libraries.")
@click.pass_context
def run(ctx, model_path, model_name, engine, model_type, device, runtime_config, scheduler_config, cache_dir, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold, tool_call_parser, host, port, use_api_key, verbose):
    """- Add a model configuration, start the server, and load the model in one step.

    Combines 'openarc add' + 'openarc serve start' + 'openarc load'.

    Examples:
        openarc run --model-name my-model --model-path /path/to/model --engine ovgenai --model-type llm --device AUTO
        openarc run -mn chatbot -m ./models/Qwen3-0.6B -en ovgenai -mt llm -d GPU --port 8080
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

    parsed_scheduler_config = {}
    if scheduler_config:
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

    # Build load config (same structure as add command)
    load_config = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type,
        "engine": engine,
        "device": device,
        "runtime_config": parsed_runtime_config,
        "scheduler_config": parsed_scheduler_config,
    }

    if cache_dir:
        load_config["cache_dir"] = cache_dir

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

    # Step 1: Save model config to JSON (like 'openarc add')
    ctx.obj.server_config.save_model_config(model_name, load_config)
    console.print(f"[green]Model configuration saved:[/green] {model_name}")

    # Step 2: Save server config for host/port
    config_path = ctx.obj.server_config.save_server_config(host, port)
    console.print(f"[dim]Configuration saved to: {config_path}[/dim]")

    # Step 3: Set API key flag if requested
    if use_api_key:
        if not os.getenv("OPENARC_API_KEY"):
            console.print("[red]Error: You chose to require an API key but OPENARC_API_KEY has not been set.[/red]")
            raise SystemExit(1)
        os.environ["OPENARC_API_KEY_REQUIRED"] = "true"
    else:
        os.environ["OPENARC_API_KEY_REQUIRED"] = "false"

    # Step 4: Start the server in a background thread, then load the model
    from ..modules.launch_server import start_server

    console.print(f"[green]Starting OpenArc server on {host}:{port}[/green]")
    console.print(f"[blue]Loading model:[/blue] {model_name}")

    # Build the base URL for loading
    base_url = f"http://{host}:{port}"

    def load_model_after_startup():
        """Wait for server to be ready, then POST the model config."""
        api_key_header = {}
        if use_api_key:
            api_key_header = {"X-API-Key": os.getenv("OPENARC_API_KEY", "")}

        max_retries = 30
        for i in range(max_retries):
            try:
                resp = requests.get(f"{base_url}/openarc/version", headers=api_key_header, timeout=2)
                if resp.status_code == 200:
                    console.print("[cyan]Server is ready[/cyan]")

                    # POST the load request
                    try:
                        console.print("[cyan]...loading model[/cyan]")
                        response = requests.post(
                            f"{base_url}/openarc/load",
                            json=load_config,
                            headers={**api_key_header},
                        )
                        if response.status_code == 200:
                            console.print(f"[green]{model_name} loaded![/green]")
                        else:
                            console.print(f"[red]Error loading model ({response.status_code}):[/red] {response.text}")
                    except requests.exceptions.RequestException as e:
                        console.print(f"[red]Failed to load model:[/red] {e}")
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(1)

        console.print("[red]Server did not become ready in time.[/red]")

    # Start server thread
    server_thread = threading.Thread(target=start_server, kwargs={"host": host, "port": port, "verbose": verbose}, daemon=True)
    server_thread.start()

    # Run the load-in-background logic in main thread (so we stay alive)
    load_model_after_startup()
