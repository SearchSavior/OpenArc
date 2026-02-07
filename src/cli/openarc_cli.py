#!/usr/bin/env python3
"""
OpenArc CLI Tool - Command-line interface for OpenArc server operations.
"""
import json
import os
import uuid
from pathlib import Path

import requests
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
 
click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
click.rich_click.STYLE_COMMANDS_TABLE_BORDER_STYLE = "red"
click.rich_click.STYLE_COMMANDS_TABLE_ROW_STYLES = ["magenta", "yellow", "cyan", "green"]

console = Console()

def validate_model_path(model_path):
    """
    Validate that model_path contains OpenVINO model files.
    Checks for at least one file with "_model.bin" and one file with "_model.xml" in filename.
    Returns True if valid, False otherwise.
    """
    path = Path(model_path)
    
    # Resolve the path
    if not path.exists():
        return False
    
    # Determine search directory - if path is a file, use its parent; if directory, use it
    if path.is_file():
        search_dir = path.parent
    else:
        search_dir = path
    
    # Check for required files
    has_bin = False
    has_xml = False
    
    try:
        for file_path in search_dir.rglob("*"):
            if file_path.is_file():
                filename = file_path.name
                if "_model.bin" in filename:
                    has_bin = True
                if "_model.xml" in filename:
                    has_xml = True
                if has_bin and has_xml:
                    return True
    except (OSError, PermissionError):
        return False
    
    return False

class CLIContext:
    """Context object for lazy-loading heavy dependencies."""
    __slots__ = ('_server_config', '_benchmark_db')
    
    def __init__(self):
        self._server_config = None
        self._benchmark_db = None
    
    @property
    def server_config(self):
        """Lazy-load ServerConfig only when needed."""
        if self._server_config is None:
            from .server_config import ServerConfig
            self._server_config = ServerConfig()
        return self._server_config
    
    @property
    def benchmark_db(self):
        """Lazy-load BenchmarkDB only when needed."""
        if self._benchmark_db is None:
            from .benchmark import BenchmarkDB
            self._benchmark_db = BenchmarkDB()
        return self._benchmark_db


class OpenArcCLI:
    def __init__(self, base_url=None, api_key=None, server_config=None):
        if base_url is None and server_config is not None:
            base_url = server_config.get_base_url()
        self.base_url = base_url
        self.api_key = api_key or os.getenv('OPENARC_API_KEY')
        
    def get_headers(self):
        """Get headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

class ColoredAsciiArtGroup(click.RichGroup):
    """Custom Click group with cached ASCII art banner for performance."""
    
    # Cache ASCII art as class attribute (built once, reused forever)
    _ascii_art_cache = None
    
    @classmethod
    def _build_ascii_art(cls) -> Text:
        """Build and cache the ASCII art banner."""
        if cls._ascii_art_cache is None:
            # Build entire ASCII art in one go for better performance
            ascii_lines = [
                (" _____                   ___           \n", "blue"),
                ("|  _  |                 / _ \\          \n", "blue"),
                ("| | | |_ __   ___ _ __ / /_\\ \\_ __ ___ \n", "blue"),
                ("| | | | '_ \\ / _ \\ '_ \\|  _  | '__/ __|\n", "blue"),
                ("\\ \\_/ / |_) |  __/ | | | | | | | | (__ \n", "blue"),
                (" \\___/| .__/ \\___|_| |_\\_| |_/_|  \\___|\n", "blue"),
                ("      | |                              \n", "blue"),
                ("      |_|                              \n", "blue"),
                (" \n", "white"),
                (" Making AI go brr since 2025   \n", "white"),
            ]
            
            art = Text()
            for line, style in ascii_lines:
                art.append(line, style=style)
            
            cls._ascii_art_cache = art
        
        return cls._ascii_art_cache
    
    def get_help(self, ctx):
        """Display help with pre-cached ASCII art banner."""
        console.print(self._build_ascii_art())
        return super().get_help(ctx)

@click.group(cls=ColoredAsciiArtGroup)
@click.pass_context
def cli(ctx):
    """
    Use this application to interface with the OpenArc server.
    
    Features:

    • Start the OpenArc server.
    
    • Load models into the OpenArc server.
    
    • List models from saved configurations.

    • Check the status of loaded models.

    • Unload models.

    • Benchmark model performance.

    • Query device properties.

    • Query installed devices.


    To get started add --help to one of the commands below to view its documentation.
    """
    ctx.ensure_object(CLIContext)

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
    type=click.Choice(['llm', 'vlm', 'whisper', 'kokoro', 'emb', 'rerank']),
    required=True,
    help='Model type (llm, vlm, whisper, kokoro, emb, rerank)')
@click.option('--device', '--d',
    required=True,
    help='Device(s) to load the model on.')
@click.option("--runtime-config", "--rtc",
    default=None,
    help='OpenVINO runtime configuration as JSON string (e.g., \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\').')
@click.option('--vlm-type', '--vt',
    type=click.Choice(['internvl2', 'llava15', 'llavanext', 'minicpmv26', 'phi3vision', 'phi4mm', 'qwen2vl', 'qwen25vl', 'gemma3']),
    required=False,
    default=None,
    help='Vision model type. Used to map correct vision tokens.')
@click.option('--draft-model-path', '--dmp',
    required=False,
    default=None,
    help='Path to draft model for speculative decoding.')
@click.option('--draft-device', '--dd',
    required=False,
    default=None,
    help='Device for draft model (e.g., CPU, GPU).')
@click.option('--num-assistant-tokens', '--nat',
    required=False,
    default=None,
    type=int,
    help='Number of tokens draft model generates per step (typically 2-5).')
@click.option('--assistant-confidence-threshold', '--act',
    required=False,
    default=None,
    type=float,
    help='Confidence threshold for accepting draft tokens (typically 0.3-0.5).')
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, vlm_type, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold):
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
    
    # Build and save configuration
    load_config = {
        "model_name": model_name,
        "model_path": model_path,  
        "model_type": model_type,  
        "engine": engine,    
        "device": device,
        "runtime_config": parsed_runtime_config,
        "vlm_type": vlm_type if vlm_type else None
    }
    
    # Add speculative decoding options if provided
    if draft_model_path:
        load_config["draft_model_path"] = draft_model_path
    if draft_device:
        load_config["draft_device"] = draft_device
    if num_assistant_tokens is not None:
        load_config["num_assistant_tokens"] = num_assistant_tokens
    if assistant_confidence_threshold is not None:
        load_config["assistant_confidence_threshold"] = assistant_confidence_threshold
    
    ctx.obj.server_config.save_model_config(model_name, load_config)
    console.print(f"[green]Model configuration saved:[/green] {model_name}")
    console.print(f"[dim]Use 'openarc load {model_name}' to load this model.[/dim]")

@cli.command()
@click.argument('model_names', nargs=-1, required=True)
@click.pass_context
def load(ctx, model_names):
    """- Load one or more models from saved configuration.
    
    Examples:
        openarc load model1
        openarc load Dolphin-X1 kokoro whisper
    """
    cli_instance = OpenArcCLI(server_config=ctx.obj.server_config)
    
    model_names = list(model_names)
    
    # Track results
    successful_loads = []
    failed_loads = []
    
    # Start loading queue
    if len(model_names) > 1:
        console.print(f"[blue]Starting load queue...[/blue] ({len(model_names)} models)\n")
    
    # Load each model
    for idx, name in enumerate(model_names, 1):
        # Show progress indicator for multiple models
        if len(model_names) > 1:
            console.print(f"[cyan]({idx}/{len(model_names)})[/cyan] [blue]loading[/blue] {name}")
        else:
            console.print(f"[blue]loading[/blue] {name}")
        
        # Get saved configuration
        saved_config = ctx.obj.server_config.get_model_config(name)
        
        if not saved_config:
            console.print(f"[red]Model configuration not found:[/red] {name}")
            console.print("[yellow]Tip: Use 'openarc list' to see saved configurations.[/yellow]\n")
            failed_loads.append(name)
            continue
        
        load_config = saved_config.copy()
        
        # Validate model path
        model_path = load_config.get('model_path')
        if model_path and not validate_model_path(model_path):
            console.print(f"[red]Model file check failed! {model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
            failed_loads.append(name)
            continue
        
        # Make API request to load the model
        url = f"{cli_instance.base_url}/openarc/load"
        
        try:
            console.print("[cyan]...working[/cyan]")
            response = requests.post(url, json=load_config, headers=cli_instance.get_headers())
            
            if response.status_code == 200:
                console.print(f"[green]{name} loaded![/green]\n")
                successful_loads.append(name)
            else:
                console.print(f"[red]error: {response.status_code}[/red]")
                console.print(f"[red]Response:[/red] {response.text}\n")
                failed_loads.append(name)
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed:[/red] {e}\n")
            failed_loads.append(name)
    
    # Summary
    console.print("─" * 60)
    if successful_loads and not failed_loads:
        console.print(f"[green]All models loaded![/green] ({len(successful_loads)}/{len(model_names)})")
    elif successful_loads and failed_loads:
        console.print(f"[yellow]Partial success:[/yellow] {len(successful_loads)}/{len(model_names)} models loaded")
        console.print(f"   [green]✓ Loaded:[/green] {', '.join(successful_loads)}")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_loads)}")
    else:
        console.print(f"[red]All models failed to load![/red] (0/{len(model_names)})")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_loads)}")
    
    console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
    
    # Exit with error code if any loads failed
    if failed_loads:
        ctx.exit(1)

@cli.command()
@click.argument('model_names', nargs=-1, required=True)
@click.pass_context
def unload(ctx, model_names):
    """
    - Unload one or more models from registry and memory.
    
    Examples:
        openarc unload model1
        openarc unload Dolphin-X1 kokoro whisper
    """
    cli_instance = OpenArcCLI(server_config=ctx.obj.server_config)

    model_names = list(model_names)
    
    # Track results
    successful_unloads = []
    failed_unloads = []
    
    # Start unloading queue
    if len(model_names) > 1:
        console.print(f"[blue]Starting unload queue...[/blue] ({len(model_names)} models)\n")
    
    # Unload each model
    for idx, name in enumerate(model_names, 1):
        # Show progress indicator for multiple models
        if len(model_names) > 1:
            console.print(f"[cyan]({idx}/{len(model_names)})[/cyan] [blue]unloading[/blue] {name}")
        else:
            console.print(f"[blue]unloading[/blue] {name}")
        
        url = f"{cli_instance.base_url}/openarc/unload"
        payload = {"model_name": name}
        
        try:
            console.print("[cyan]...working[/cyan]")
            response = requests.post(url, json=payload, headers=cli_instance.get_headers())
            
            if response.status_code == 200:
                result = response.json()
                message = result.get('message', f"Model '{name}' unloaded successfully")
                console.print(f"[green]{message}[/green]\n")
                successful_unloads.append(name)
            else:
                console.print(f"[red]error: {response.status_code}[/red]")
                console.print(f"[red]Response:[/red] {response.text}\n")
                failed_unloads.append(name)
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed:[/red] {e}\n")
            failed_unloads.append(name)
    
    # Summary
    console.print("─" * 60)
    if successful_unloads and not failed_unloads:
        console.print(f"[green]All models unloaded![/green] ({len(successful_unloads)}/{len(model_names)})")
    elif successful_unloads and failed_unloads:
        console.print(f"[yellow]Partial success:[/yellow] {len(successful_unloads)}/{len(model_names)} models unloaded")
        console.print(f"   [green]✓ Unloaded:[/green] {', '.join(successful_unloads)}")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_unloads)}")
    else:
        console.print(f"[red]All models failed to unload![/red] (0/{len(model_names)})")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_unloads)}")
    
    console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
    
    # Exit with error code if any unloads failed
    if failed_unloads:
        ctx.exit(1)

@cli.command("list")
@click.option('--model-name','--mn', help='Model name to remove (used with --remove/--rm).')
@click.option('--remove', '--rm', is_flag=True, help='Remove a model configuration.')
@click.pass_context
def list_configs(ctx, remove, model_name):
    """- List saved model configurations.
       
       - Remove a model configuration."""
    if remove:
        if not model_name:
            console.print("[red]Error:[/red] --model-name is required when using --remove")

            ctx.exit(1)
        
        # Check if model exists before trying to remove
        if not ctx.obj.server_config.model_exists(model_name):
            console.print(f"{model_name}[red] not found:[/red]")
            console.print("[yellow]Use 'openarc list' to see available configurations.[/yellow]")
            ctx.exit(1)
        
        # Remove the configuration
        if ctx.obj.server_config.remove_model_config(model_name):
            console.print(f"[green]Model configuration removed:[/green] {model_name}")
        else:
            console.print(f"[red]Failed to remove model configuration:[/red] {model_name}")
            ctx.exit(1)
        return
    
    models = ctx.obj.server_config.get_all_models()
    
    if not models:
        console.print("[yellow]No model configurations found.[/yellow]")
        console.print("[dim]Use 'openarc add --help' to see how to save configurations.[/dim]")
        return
    
    console.print(f"[blue]Saved Model Configurations ({len(models)}):[/blue]\n")
    
    for model_name, model_config in models.items():
        # Create a table for each model configuration
        config_table = Table(show_header=False, box=None, pad_edge=False)
        

        config_table.add_row("model_name", f"[cyan]{model_name}[/cyan]")
        config_table.add_row("model_path", f"[yellow]{model_config.get('model_path')}[/yellow]")
        config_table.add_row("device", f"[blue]{model_config.get('device')}[/blue]")
        config_table.add_row("engine", f"[green]{model_config.get('engine')}[/green]")
        config_table.add_row("model_type", f"[magenta]{model_config.get('model_type')}[/magenta]")

        # Display optional fields when available
        if model_config.get('draft_model_path'):
            config_table.add_row("draft_model_path", f"[red]{model_config.get('draft_model_path')}[/red]")
        if model_config.get('draft_device'):
            config_table.add_row("draft_device", f"[red]{model_config.get('draft_device')}[/red]")
        if model_config.get('num_assistant_tokens') is not None:
            config_table.add_row("num_assistant_tokens", f"[red]{model_config.get('num_assistant_tokens')}[/red]")
        if model_config.get('assistant_confidence_threshold') is not None:
            config_table.add_row("assistant_confidence_threshold", f"[red]{model_config.get('assistant_confidence_threshold')}[/red]")


        rtc = model_config.get('runtime_config', {})
        if rtc:
            config_table.add_row("", "")
            config_table.add_row(Text("runtime_config", style="bold underline yellow"), "")
            for key, value in rtc.items():
                config_table.add_row(f"  {key}", f"[dim]{value}[/dim]")
        
        panel = Panel(
            config_table,
            border_style="green"
        )
        console.print(panel)
    
    console.print("\n[dim]To load saved configurations: openarc load <model_name> [model_name2 ...][/dim]")
    console.print("[dim]To remove a configuration: openarc list --remove --model-name <model_name>[/dim]")

@cli.command()
@click.pass_context
def status(ctx):
    """- GET Status of loaded models."""
    cli_instance = OpenArcCLI(server_config=ctx.obj.server_config)
    
    url = f"{cli_instance.base_url}/openarc/status"
    
    try:
        console.print("[blue]Getting model status...[/blue]")
        response = requests.get(url, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            total_models = result.get("total_loaded_models", 0)
            
            if not models:
                console.print("[yellow]No models currently loaded.[/yellow]")
            else:
                # Create a table for all models
                status_table = Table(title=f"Loaded Models ({total_models})")
                status_table.add_column("model_name", style="cyan", width=20)
                status_table.add_column("device", style="blue", width=10)
                status_table.add_column("model_type", style="magenta", width=15)
                status_table.add_column("engine", style="green", width=10)
                status_table.add_column("status", style="yellow", width=10)
                status_table.add_column("time_loaded", style="dim", width=20)
                
                for model in models:
                    model_name = model.get("model_name")
                    device = model.get("device")
                    model_type = model.get("model_type")
                    engine = model.get("engine")
                    status = model.get("status")
                    time_loaded = model.get("time_loaded")
                    
                    status_table.add_row(
                        model_name,
                        device,
                        model_type,
                        engine,
                        status,
                        time_loaded
                    )
                
                console.print(status_table)
                console.print(f"\n[green]Total models loaded: {total_models}[/green]")
            
        else:
            console.print(f"[red]Error getting status: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Request failed:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.argument('model_name')
@click.option('--input_tokens', '--p', multiple=True, default=['512'],
              help='Number of prompt tokens. Can be comma-separated (e.g., --p 16,32) or specified multiple times (e.g., -p 16 -p 32). Default: 512')
@click.option('--max_tokens', '--n', multiple=True, default=['128'],
              help='Number of tokens to generate. Can be comma-separated or specified multiple times. Default: 128')
@click.option('--runs', '--r', default=5, type=int,
              help='Number of times to repeat each benchmark. Default: 5')
@click.option('--temperature', '--temp', default=None, type=float,
              help='Sampling temperature (default: 1.0)')
@click.option('--top-k', '--k', default=None, type=int,
              help='Top-k sampling (default: 50)')
@click.option('--top-p', '--p-nucleus', default=None, type=float,
              help='Top-p nucleus sampling (default: 1.0)')
@click.option('--repetition-penalty', '--rep', default=None, type=float,
              help='Repetition penalty (default: 1.0)')
@click.pass_context
def bench(ctx, model_name, input_tokens, max_tokens, runs, temperature, top_k, top_p, repetition_penalty):
    """- Benchmark inference with pseudo-random input tokens.
    
    Examples:
        openarc bench Dolphin-X1
        openarc bench Dolphin-X1 --p 512 --n 128 -r 10
        openarc bench Dolphin-X1 --p 16,32,64 --n 128,256
        openarc bench Dolphin-X1 -p 16 -p 32 -n 128 -n 256
    """
    cli_instance = OpenArcCLI(server_config=ctx.obj.server_config)
    
    # Parse input_tokens and max_tokens (handle comma-separated and multiple invocations)
    p_values = []
    for pt in input_tokens:
        p_values.extend([int(x.strip()) for x in pt.split(',')])
    
    n_values = []
    for nt in max_tokens:
        n_values.extend([int(x.strip()) for x in nt.split(',')])
    
    # Check if model exists
    try:
        console.print("[cyan]working...[/cyan]\n")
        models_url = f"{cli_instance.base_url}/v1/models"
        models_response = requests.get(models_url, headers=cli_instance.get_headers())
        
        if models_response.status_code != 200:
            console.print(f"[red]Failed to get model list: {models_response.status_code}[/red]")
            ctx.exit(1)
        
        models_data = models_response.json()
        available_models = [m['id'] for m in models_data.get('data', [])]
        
        if model_name not in available_models:
            console.print(f"[red]'{model_name}' not found in loaded models[/red]")
            console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
            console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
            ctx.exit(1)
        
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Request failed:[/red] {e}")
        ctx.exit(1)
    
    # Get model path from config to generate input tokens
    model_config = ctx.obj.server_config.get_model_config(model_name)
    if not model_config:
        console.print(f"[red]Model configuration not found for '{model_name}'[/red]")
        console.print("[yellow]Cannot generate random tokens without model path.[/yellow]")
        console.print("[blue]Use 'openarc list' to see saved configurations.[/blue]")
        ctx.exit(1)
    
    model_path = model_config.get('model_path')
    if not model_path:
        console.print("[red]model_path not found in configuration[/red]")
        ctx.exit(1)
    
    # Validate model path
    if not validate_model_path(model_path):
        console.print(f"[red]Model file check failed! {model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
        ctx.exit(1)
    
    # Run benchmarks
    console.print(f"input tokens: {p_values}")
    console.print(f"max tokens:   {n_values}")
    console.print(f"runs: {runs}\n")
    
    # Lazy load OpenArcBenchmarks
    from .benchmark import OpenArcBenchmarks
    
    # Generate unique run_id for this benchmark session
    run_id = str(uuid.uuid4())
    
    total_runs = len(p_values) * len(n_values) * runs
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running... (0/{total_runs})", total=total_runs)
        
        run_count = 0
        for p in p_values:
            for n in n_values:
                for r in range(runs):
                    run_count += 1
                    progress.update(task, description=f"[dim]benching...[/dim] ({run_count}/{total_runs}) [p={p}, n={n}, r={r+1}/{runs}]")
                    
                    try:
                        # Generate random input tokens
                        input_ids = OpenArcBenchmarks.random_input_ids(model_path, p)
                        
                        # Make benchmark request
                        bench_url = f"{cli_instance.base_url}/openarc/bench"
                        payload = {
                            "model": model_name,
                            "input_ids": input_ids,
                            "max_tokens": n
                        }

                        # Add optional parameters if provided
                        if temperature is not None:
                            payload["temperature"] = temperature
                        if top_k is not None:
                            payload["top_k"] = top_k
                        if top_p is not None:
                            payload["top_p"] = top_p
                        if repetition_penalty is not None:
                            payload["repetition_penalty"] = repetition_penalty

                        bench_response = requests.post(
                            bench_url,
                            headers=cli_instance.get_headers(),
                            json=payload
                        )
                        
                        if bench_response.status_code != 200:
                            console.print(f"\n[red]Benchmark request failed: {bench_response.status_code}[/red]")
                            console.print(f"[red]Response:[/red] {bench_response.text}")
                            continue
                        
                        metrics = bench_response.json().get('metrics', {})
                        
                        # Store individual result
                        result = {
                            'p': p,
                            'n': n,
                            'run': r + 1,
                            'ttft': metrics.get('ttft (s)', 0),
                            'tpot': metrics.get('tpot (ms)', 0),
                            'prefill_throughput': metrics.get('prefill_throughput (tokens/s)', 0),
                            'decode_throughput': metrics.get('decode_throughput (tokens/s)', 0),
                            'decode_duration': metrics.get('decode_duration (s)', 0),
                            'input_token': metrics.get('input_token', 0),
                            'new_token': metrics.get('new_token', 0),
                            'total_token': metrics.get('total_token', 0),
                        }
                        results.append(result)
                        
                        # Save result to database
                        ctx.obj.benchmark_db.save_result(model_name, result, run_id)
                        
                    except Exception as e:
                        console.print(f"\n[yellow]Error in run {r+1}: {e}[/yellow]")
                        continue
                    
                    progress.advance(task)
    
    # Display results
    console.print("\n")
    
    if not results:
        console.print("[red]No benchmark results collected![/red]")
        ctx.exit(1)
    
    
    
    
    model_path_name = Path(model_path).name
    console.print(f"\n[blue]{model_path_name}[/blue]\n")

    # Create results table with visible lines
    results_table = Table(show_header=True, header_style="bold")
    results_table.add_column("[cyan]run[/cyan]", justify="right")
    results_table.add_column("[cyan]p[/cyan]", justify="right")
    results_table.add_column("[cyan]n[/cyan]", justify="right")
    results_table.add_column("[cyan]ttft(s)[/cyan]", justify="right")
    results_table.add_column("[cyan]tpot(ms)[/cyan]", justify="right")
    results_table.add_column("[cyan]prefill(t/s)[/cyan]", justify="right")
    results_table.add_column("[cyan]decode(t/s)[/cyan]", justify="right")
    results_table.add_column("[cyan]duration(s)[/cyan]", justify="right")

    
    for result in results:
        results_table.add_row(
            str(result['run']),
            str(result['p']),
            str(result['n']),
            f"{result['ttft']:.2f}",
            f"{result['tpot']:.2f}",
            f"{result['prefill_throughput']:.1f}",
            f"{result['decode_throughput']:.1f}",
            f"{result['decode_duration']:.2f}"
            )
    
    console.print(results_table)
    

    console.print(f"[dim]Total: {len(results)} runs[/dim]")

@cli.group()
@click.pass_context
def tool(ctx):
    """- Utility scripts."""
    pass

@tool.command('device-props')
@click.pass_context
def device_properties(ctx):
    """
    - Query OpenVINO device properties for all available devices.
    """
    
    try:
        from .device_query import DeviceDataQuery
        console.print("[blue]Querying device data for all devices...[/blue]")
        device_query = DeviceDataQuery()
        available_devices = device_query.get_available_devices()
        
        console.print(f"\n[green]Available Devices ({len(available_devices)}):[/green]")
        
        if not available_devices:
            console.print("[red]No devices found![/red]")
            ctx.exit(1)
        
        for device in available_devices:
            # Create a panel for each device
            properties = device_query.get_device_properties(device)
            properties_text = "\n".join([f"{key}: {value}" for key, value in properties.items()])
            
            panel = Panel(
                properties_text,
                title=f"Device: {device}",
                title_align="left",
                border_style="blue"
            )
            console.print(panel)
        
        console.print(f"\n[green]Found {len(available_devices)} device(s)[/green]")
        
    except Exception as e:
        console.print(f"[red]Error querying device data:[/red] {e}")
        ctx.exit(1)

@tool.command('device-detect')
@click.pass_context
def device_detect(ctx):
    """
    - Detect available OpenVINO devices.
    """
    
    try:
        from .device_query import DeviceDiagnosticQuery
        console.print("[blue]Detecting OpenVINO devices...[/blue]")
        diagnostic = DeviceDiagnosticQuery()
        available_devices = diagnostic.get_available_devices()
        
        table = Table()
        table.add_column("Index", style="cyan", width=2)
        table.add_column("Device", style="green")
        
        if not available_devices:
            console.print("[red] Sanity test failed: No OpenVINO devices found! Maybe check your drivers?[/red]")
            ctx.exit(1)
        
        for i, device in enumerate(available_devices, 1):
            table.add_row(str(i), device)
        
        console.print(table)
        console.print(f"\n[green] Sanity test passed: found {len(available_devices)} device(s)[/green]")
            
    except Exception as e:
        console.print(f"[red]Sanity test failed: No OpenVINO devices found! Maybe check your drivers?[/red] {e}")
        ctx.exit(1)

@cli.group()
def serve():
    """
    - Start the OpenArc server.
    """
    pass

@serve.command("start")
@click.option("--host", type=str, default="0.0.0.0", show_default=True,
              help="""
              - Host to bind the server to
              """)
@click.option("--openarc-port", 
              type=int, 
              default=8000, 
              show_default=True,
              help="""
              - Port to bind the server to
              """)
@click.option("--load-models", "--lm",
              required=False,
              help="Load models on startup. Specify once followed by space-separated model names.")
@click.argument('startup_models', nargs=-1, required=False)
@click.pass_context
def start(ctx, host, openarc_port, load_models, startup_models):
    """
    - 'start' reads --host and --openarc-port from config or defaults to 0.0.0.0:8000
    
    Examples:
        openarc serve start
        openarc serve start --load-models model1 model2
        openarc serve start --lm Dolphin-X1 kokoro whisper
    """
    # Save server configuration for other CLI commands to use
    config_path = ctx.obj.server_config.save_server_config(host, openarc_port)
    console.print(f"[dim]Configuration saved to: {config_path}[/dim]")
    
    # Handle startup models
    models_to_load = []
    if load_models:
        models_to_load.append(load_models)
    if startup_models:
        models_to_load.extend(startup_models)
    
    if models_to_load:
        saved_model_names = ctx.obj.server_config.get_model_names()
        missing = [m for m in models_to_load if m not in saved_model_names]
        
        if missing:
            console.print("[yellow]Warning: Models not in config (will be skipped):[/yellow]")
            for m in missing:
                console.print(f"   • {m}")
            console.print("[dim]Use 'openarc list' to see saved configurations.[/dim]\n")
        
        os.environ["OPENARC_STARTUP_MODELS"] = ",".join(models_to_load)
        console.print(f"[blue]Models to load on startup:[/blue] {', '.join(models_to_load)}\n")
    
    console.print(f"[green]Starting OpenArc server on {host}:{openarc_port}[/green]")
    from .launch_server import start_server
    start_server(host=host, openarc_port=openarc_port)


if __name__ == "__main__":
    cli()



