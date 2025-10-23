#!/usr/bin/env python3
"""
OpenArc CLI Tool - Command-line interface for OpenArc server operations.
"""
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import requests
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.server.launch import start_server
from src.cli.device_query import DeviceDataQuery, DeviceDiagnosticQuery
from src.cli.openarc_bench import num_input_ids

click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
click.rich_click.STYLE_COMMANDS_TABLE_BORDER_STYLE = "red"
click.rich_click.STYLE_COMMANDS_TABLE_ROW_STYLES = ["magenta", "yellow", "cyan", "green"]

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "openarc-config.json"
BENCH_DB = PROJECT_ROOT / "openarc-bench.db"

def init_bench_db():
    """Initialize benchmark database and create table if it doesn't exist."""
    conn = sqlite3.connect(BENCH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            bench_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            max_tokens INTEGER NOT NULL,
            run_number INTEGER NOT NULL,
            ttft_s TEXT,
            tpot_ms TEXT,
            prefill_throughput_tokens_s TEXT,
            decode_throughput_tokens_s TEXT,
            decode_duration_s TEXT,
            input_token_count TEXT,
            new_token_count TEXT,
            total_token_count TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_bench_result(model_name: str, result: dict):
    """Save a single benchmark result to the database."""
    conn = sqlite3.connect(BENCH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO benchmark_results (
            timestamp, model_name, input_tokens, max_tokens, run_number,
            ttft_s, tpot_ms, prefill_throughput_tokens_s, decode_throughput_tokens_s,
            decode_duration_s, input_token_count, new_token_count, total_token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_name,
        result['p'],
        result['n'],
        result['run'],
        str(result['ttft']),
        str(result['tpot']),
        str(result['prefill_throughput']),
        str(result['decode_throughput']),
        str(result['decode_duration']),
        result['input_token'],
        result['new_token'],
        result['total_token']
    ))
    
    conn.commit()
    conn.close()

def save_cli_config(host: str, port: int):
    """Save server configuration to JSON config file."""
    config = load_full_config()  # Load existing config first
    config.update({
        "server": {
            "host": host,
            "port": port
        },
        "created_by": "openarc-cli",
        "version": "1.0"
    })
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    console.print(f"📝 [dim]Configuration saved to: {CONFIG_FILE}[/dim]")

def save_model_config(model_name: str, load_config: dict):
    """Save model configuration to JSON config file."""
    config = load_full_config()
    
    if "models" not in config:
        config["models"] = {}
    
    config["models"][model_name] = load_config
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    console.print(f"💾 [green]Model configuration saved:[/green] {model_name}")

def load_full_config():
    """Load full configuration from JSON config file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config if config else {}
        except (json.JSONDecodeError, FileNotFoundError):
            console.print(f"[yellow]Warning: Could not read config file {CONFIG_FILE}[/yellow]")
    
    return {}

def get_model_config(model_name: str):
    """Get model configuration by name."""
    config = load_full_config()
    models = config.get("models", {})
    return models.get(model_name)

def remove_model_config(model_name: str):
    """Remove model configuration by name."""
    config = load_full_config()
    models = config.get("models", {})
    
    if model_name not in models:
        return False
    
    del models[model_name]
    config["models"] = models
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    return True

def load_cli_config():
    """Load server configuration from YAML config file."""
    config = load_full_config()
    if config and "server" in config:
        return config["server"]
    
    return {"host": "localhost", "port": 8000}  # defaults

class OpenArcCLI:
    def __init__(self, base_url=None, api_key=None):
        if base_url is None:
            config = load_cli_config()
            base_url = f"http://{config['host']}:{config['port']}"
        self.base_url = base_url
        self.api_key = api_key or os.getenv('OPENARC_API_KEY')
        
    def get_headers(self):
        """Get headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

class ColoredAsciiArtGroup(click.RichGroup):
    def get_help(self, ctx):
        console = Console()
        art = Text()
        art.append(" _____                   ___           \n", style="blue")
        art.append("|  _  |                 / _ \\          \n", style="blue")
        art.append("| | | |_ __   ___ _ __ / /_\\ \\_ __ ___ \n", style="blue")
        art.append("| | | | '_ \\ / _ \\ '_ \\|  _  | '__/ __|\n", style="blue")
        art.append("\\ \\_/ / |_) |  __/ | | | | | | | | (__ \n", style="blue")
        art.append(" \\___/| .__/ \\___|_| |_\\_| |_/_|  \\___|\n", style="blue")
        art.append("      | |                              \n", style="white")
        art.append("      |_|                              \n", style="white")
        art.append(" \n", style="white")
        art.append(" The CLI application   \n", style="white")
        console.print(art)
        return super().get_help(ctx)

@click.group(cls=ColoredAsciiArtGroup)
def cli():
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
    help='OpenVINO runtime configuration (e.g., performance hints). These are checked serverside at runtime.')
@click.option('--vlm-type', '--vt',
    type=click.Choice(['internvl2', 'llava15', 'llavanext', 'minicpmv26', 'phi3vision', 'phi4mm', 'qwen2vl', 'qwen25vl', 'gemma3']),
    required=False,
    default=None,
    help='Vision model type. Used to map correct vision tokens.')
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, vlm_type):
    """- Add a model configuration to the config file."""
    
    # Build and save configuration
    load_config = {
        "model_name": model_name,
        "model_path": model_path,  
        "model_type": model_type,  
        "engine": engine,    
        "device": device,
        "runtime_config": runtime_config if runtime_config else {},
        "vlm_type": vlm_type if vlm_type else None
    }
    
    save_model_config(model_name, load_config)
    console.print(f"✅ [green]Saved configuration for:[/green] {model_name}")
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
    cli_instance = OpenArcCLI()
    
    model_names = list(model_names)
    
    # Track results
    successful_loads = []
    failed_loads = []
    
    # Start loading queue
    if len(model_names) > 1:
        console.print(f"🚀 [blue]Starting load queue...[/blue] ({len(model_names)} models)\n")
    
    # Load each model
    for idx, name in enumerate(model_names, 1):
        # Show progress indicator for multiple models
        if len(model_names) > 1:
            console.print(f"[cyan]({idx}/{len(model_names)})[/cyan] [blue]loading[/blue] {name}")
        else:
            console.print(f"[blue]loading[/blue] {name}")
        
        # Get saved configuration
        saved_config = get_model_config(name)
        
        if not saved_config:
            console.print(f"❌ [red]Model configuration not found:[/red] {name}")
            console.print("[yellow]Tip: Use 'openarc list' to see saved configurations.[/yellow]\n")
            failed_loads.append(name)
            continue
        
        load_config = saved_config.copy()
        
        # Make API request to load the model
        url = f"{cli_instance.base_url}/openarc/load"
        
        try:
            console.print("[cyan]...working[/cyan]")
            response = requests.post(url, json=load_config, headers=cli_instance.get_headers())
            
            if response.status_code == 200:
                console.print(f"✅ [green]{name} loaded![/green]\n")
                successful_loads.append(name)
            else:
                console.print(f"❌ [red]error: {response.status_code}[/red]")
                console.print(f"[red]Response:[/red] {response.text}\n")
                failed_loads.append(name)
                
        except requests.exceptions.RequestException as e:
            console.print(f"❌ [red]Request failed:[/red] {e}\n")
            failed_loads.append(name)
    
    # Summary
    console.print("─" * 60)
    if successful_loads and not failed_loads:
        console.print(f"🎉 [green]All models loaded![/green] ({len(successful_loads)}/{len(model_names)})")
    elif successful_loads and failed_loads:
        console.print(f"⚠️  [yellow]Partial success:[/yellow] {len(successful_loads)}/{len(model_names)} models loaded")
        console.print(f"   [green]✓ Loaded:[/green] {', '.join(successful_loads)}")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_loads)}")
    else:
        console.print(f"❌ [red]All models failed to load![/red] (0/{len(model_names)})")
        console.print(f"   [red]✗ Failed:[/red] {', '.join(failed_loads)}")
    
    console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
    
    # Exit with error code if any loads failed
    if failed_loads:
        ctx.exit(1)

@cli.command()
@click.option('--model-name', '--mn', required=True, help='Model name to unload')
@click.pass_context
def unload(ctx, model_name):
    """
    - POST Delete a model from registry and unload from memory.
    """
    cli_instance = OpenArcCLI()

    url = f"{cli_instance.base_url}/openarc/unload"
    payload = {"model_name": model_name}
    
    try:
        console.print(f"🗑️  [blue]Unloading model:[/blue] {model_name}")
        response = requests.post(url, json=payload, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            result = response.json()
            # Handle different possible response formats
            message = result.get('message', f"Model '{model_name}' unloaded successfully")
            console.print(f"✅ [green]{message}[/green]")
        else:
            console.print(f"❌ [red]Error unloading model: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
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
            console.print("❌ [red]Error:[/red] --model-name is required when using --remove")

            ctx.exit(1)
        
        # Check if model exists before trying to remove
        existing_config = get_model_config(model_name)
        if not existing_config:
            console.print(f"❌ {model_name}[red] not found:[/red]")
            console.print("[yellow]Use 'openarc list' to see available configurations.[/yellow]")
            ctx.exit(1)
        
        # Remove the configuration
        if remove_model_config(model_name):
            console.print(f"🗑️  [green]Model configuration removed:[/green] {model_name}")
        else:
            console.print(f"❌ [red]Failed to remove model configuration:[/red] {model_name}")
            ctx.exit(1)
        return
    
    config = load_full_config()
    models = config.get("models", {})
    
    if not models:
        console.print("[yellow]No model configurations found.[/yellow]")
        console.print("[dim]Use 'openarc add --help' to see how to save configurations.[/dim]")
        return
    
    console.print(f"📋 [blue]Saved Model Configurations ({len(models)}):[/blue]\n")
    
    for model_name, model_config in models.items():
        # Create a table for each model configuration
        config_table = Table(show_header=False, box=None, pad_edge=False)
        

        config_table.add_row("model_name", f"[cyan]{model_name}[/cyan]")
        config_table.add_row("device", f"[blue]{model_config.get('device')}[/blue]")
        config_table.add_row("engine", f"[green]{model_config.get('engine')}[/green]")
        config_table.add_row("model_type", f"[magenta]{model_config.get('model_type')}[/magenta]")
        
        
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
    cli_instance = OpenArcCLI()
    
    url = f"{cli_instance.base_url}/openarc/status"
    
    try:
        console.print("📊 [blue]Getting model status...[/blue]")
        response = requests.get(url, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            total_models = result.get("total_loaded_models", 0)
            
            if not models:
                console.print("[yellow]No models currently loaded.[/yellow]")
            else:
                # Create a table for all models
                status_table = Table(title=f"📊 Loaded Models ({total_models})")
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
            console.print(f"❌ [red]Error getting status: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.argument('model_name')
@click.option('--input_tokens', '--p', multiple=True, default=['512'],
              help='Number of prompt tokens. Can be comma-separated (e.g., --p 16,32) or specified multiple times (e.g., -p 16 -p 32). Default: 512')
@click.option('--max_tokens', '--n', multiple=True, default=['128'],
              help='Number of tokens to generate. Can be comma-separated or specified multiple times. Default: 128')
@click.option('--runs', '--r', type=int, default=5,
              help='Number of times to repeat each benchmark. Default: 5')
@click.pass_context
def bench(ctx, model_name, input_tokens, max_tokens, runs):
    """- Benchmark inference with pseudo-random input tokens.
    
    Examples:
        openarc bench Dolphin-X1
        openarc bench Dolphin-X1 --p 512 --n 128 -r 10
        openarc bench Dolphin-X1 --p 16,32,64 --n 128,256
        openarc bench Dolphin-X1 -p 16 -p 32 -n 128 -n 256
    """
    # Initialize benchmark database
    init_bench_db()
    
    cli_instance = OpenArcCLI()
    
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
            console.print(f"❌ [red]Failed to get model list: {models_response.status_code}[/red]")
            ctx.exit(1)
        
        models_data = models_response.json()
        available_models = [m['id'] for m in models_data.get('data', [])]
        
        if model_name not in available_models:
            console.print(f"❌ [red]'{model_name}' not found in loaded models[/red]")
            console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
            console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
            ctx.exit(1)
        
        
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
        ctx.exit(1)
    
    # Get model path from config to generate input tokens
    model_config = get_model_config(model_name)
    if not model_config:
        console.print(f"❌ [red]Model configuration not found for '{model_name}'[/red]")
        console.print("[yellow]Cannot generate random tokens without model path.[/yellow]")
        console.print("[blue]Use 'openarc list' to see saved configurations.[/blue]")
        ctx.exit(1)
    
    model_path = model_config.get('model_path')
    if not model_path:
        console.print("❌ [red]model_path not found in configuration[/red]")
        ctx.exit(1)
    
    # Run benchmarks
    console.print(f"input tokens: {p_values}")
    console.print(f"max tokens:   {n_values}")
    console.print(f"runs: {runs}\n")
    
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
                        input_ids = num_input_ids(model_path, p)
                        
                        # Make benchmark request
                        bench_url = f"{cli_instance.base_url}/openarc/bench"
                        bench_response = requests.post(
                            bench_url,
                            headers=cli_instance.get_headers(),
                            json={
                                "model": model_name,
                                "input_ids": input_ids,
                                "max_tokens": n
                            }
                        )
                        
                        if bench_response.status_code != 200:
                            console.print(f"\n❌ [red]Benchmark request failed: {bench_response.status_code}[/red]")
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
                        save_bench_result(model_name, result)
                        
                    except Exception as e:
                        console.print(f"\n⚠️  [yellow]Error in run {r+1}: {e}[/yellow]")
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
        console.print("🔍 [blue]Querying device data for all devices...[/blue]")
        device_query = DeviceDataQuery()
        available_devices = device_query.get_available_devices()
        
        console.print(f"\n📊 [green]Available Devices ({len(available_devices)}):[/green]")
        
        if not available_devices:
            console.print("❌ [red]No devices found![/red]")
            ctx.exit(1)
        
        for device in available_devices:
            # Create a panel for each device
            properties = device_query.get_device_properties(device)
            properties_text = "\n".join([f"{key}: {value}" for key, value in properties.items()])
            
            panel = Panel(
                properties_text,
                title=f"🔹 Device: {device}",
                title_align="left",
                border_style="blue"
            )
            console.print(panel)
        
        console.print(f"\n✅ [green]Found {len(available_devices)} device(s)[/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error querying device data:[/red] {e}")
        ctx.exit(1)

@tool.command('device-detect')
@click.pass_context
def device_detect(ctx):
    """
    - Detect available OpenVINO devices.
    """
    
    try:
        console.print("🔍 [blue]Detecting OpenVINO devices...[/blue]")
        diagnostic = DeviceDiagnosticQuery()
        available_devices = diagnostic.get_available_devices()
        
        table = Table()
        table.add_column("Index", style="cyan", width=2)
        table.add_column("Device", style="green")
        
        if not available_devices:
            console.print("❌ [red] Sanity test failed: No OpenVINO devices found! Maybe check your drivers?[/red]")
            ctx.exit(1)
        
        for i, device in enumerate(available_devices, 1):
            table.add_row(str(i), device)
        
        console.print(table)
        console.print(f"\n✅ [green] Sanity test passed: found {len(available_devices)} device(s)[/green]")
            
    except Exception as e:
        console.print(f"❌ [red]Sanity test failed: No OpenVINO devices found! Maybe check your drivers?[/red] {e}")
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
def start(host, openarc_port, load_models, startup_models):
    """
    - 'start' reads --host and --openarc-port from config or defaults to 0.0.0.0:8000
    
    Examples:
        openarc serve start
        openarc serve start --load-models model1 model2
        openarc serve start --lm Dolphin-X1 kokoro whisper
    """
    # Save server configuration for other CLI commands to use
    save_cli_config(host, openarc_port)
    
    # Handle startup models
    models_to_load = []
    if load_models:
        models_to_load.append(load_models)
    if startup_models:
        models_to_load.extend(startup_models)
    
    if models_to_load:
        config = load_full_config()
        saved_models = config.get("models", {})
        missing = [m for m in models_to_load if m not in saved_models]
        
        if missing:
            console.print("⚠️  [yellow]Warning: Models not in config (will be skipped):[/yellow]")
            for m in missing:
                console.print(f"   • {m}")
            console.print("[dim]Use 'openarc list' to see saved configurations.[/dim]\n")
        
        os.environ["OPENARC_STARTUP_MODELS"] = ",".join(models_to_load)
        console.print(f"📋 [blue]Models to load on startup:[/blue] {', '.join(models_to_load)}\n")
    
    console.print(f"🚀 [green]Starting OpenArc server on {host}:{openarc_port}[/green]")
    start_server(host=host, openarc_port=openarc_port)


if __name__ == "__main__":
    cli()



