#!/usr/bin/env python3
"""
OpenArc CLI Tool - Command-line interface for OpenArc server operations.
"""
import os
import json
from pathlib import Path

import requests
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.server.launch import start_server
from src.cli.device_query import DeviceDataQuery, DeviceDiagnosticQuery

click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
click.rich_click.STYLE_COMMANDS_TABLE_BORDER_STYLE = "red"
click.rich_click.STYLE_COMMANDS_TABLE_ROW_STYLES = ["magenta", "yellow", "cyan", "green"]

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "openarc-config.json"

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
    type=click.Choice(['llm', 'vlm', 'whisper', 'kokoro']),
    required=True,
    help='Model type (llm, vlm, whisper, kokoro)')
@click.option('--device', '--d',
    required=True,
    help='Device(s) to load the model on.')
@click.option("--runtime-config", "--rtc",
    type=dict,
    default={},
    help='OpenVINO runtime configuration (e.g., performance hints). These are checked serverside at runtime.')
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config):
    """- Add a model configuration to the config file."""
    
    # Build and save configuration
    load_config = {
        "model_name": model_name,
        "model_path": model_path,  
        "model_type": model_type,  
        "engine": engine,    
        "device": device,
        "runtime_config": runtime_config if runtime_config else {}
    }
    
    save_model_config(model_name, load_config)
    console.print(f"✅ [green]Saved configuration for:[/green] {model_name}")
    console.print(f"[dim]Use 'openarc load --mn {model_name}' to load this model.[/dim]")

@cli.command()
@click.option('--model-name', '--mn',
    required=True,
    help='Model name to load from saved configuration.')
@click.pass_context
def load(ctx, model_name):
    """- Load a model from saved configuration."""
    cli_instance = OpenArcCLI()
    
    # Get saved configuration
    saved_config = get_model_config(model_name)
    
    if not saved_config:
        console.print(f"❌ [red]Model configuration not found:[/red] {model_name}")
        console.print("[yellow]Tip: Use 'openarc list' to see saved configurations, or 'openarc add' to create a new one.[/yellow]")
        ctx.exit(1)
    
    load_config = saved_config.copy()
    
    # Make API request to load the model
    url = f"{cli_instance.base_url}/openarc/load"
    
    try:
        console.print("[cyan]working...[/cyan]")
        response = requests.post(url, json=load_config, headers=cli_instance.get_headers())
        
        if response.status_code == 200:

            console.print("[green]Done![/green]")
            console.print("[dim]Use 'openarc status' to check the status of loaded models.[/dim]")
        else:
            console.print(f"❌ [red]error: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
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

@cli.command()
@click.option('--model-name','--mn', help='Model name to remove (used with --rm).')
@click.option('--rm', is_flag=True, help='Remove a model configuration.')
@click.pass_context
def list(ctx, rm, model_name):
    """- List saved model configurations.
       
       - Remove a model configuration."""
    if rm:
        if not model_name:
            console.print("❌ [red]Error:[/red] --model-name is required when using --rm")

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
    
    console.print("\n[dim]To load a saved configuration: openarc load --model-name <model_name>[/dim]")
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

@cli.group()
@click.pass_context
def tool(ctx):
    """- Utility scripts."""
    pass

@tool.command('dev-props')
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

@tool.command('dev-detect')
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

def start(host, openarc_port):
    """
    - 'start' reads --host and --openarc-port from config or defaults to 0.0.0.0:8000
    """
    # Save server configuration for other CLI commands to use
    save_cli_config(host, openarc_port)
    
    console.print(f"🚀 [green]Starting OpenArc server on {host}:{openarc_port}[/green]")
    start_server(host=host, openarc_port=openarc_port)


if __name__ == "__main__":
    cli()



