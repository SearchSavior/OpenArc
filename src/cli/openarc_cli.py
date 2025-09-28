#!/usr/bin/env python3
"""
OpenArc CLI Tool - Command-line interface for OpenArc server operations.
"""
import os
from pathlib import Path

import requests
import rich_click as click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.server.launch import start_server
from src.cli.device_query import DeviceDataQuery, DeviceDiagnosticQuery

click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_BOX = "SIMPLE"

# click.rich_click.STYLE_OPTIONS_TABLE_ROW_STYLES = ["bold", ""]
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
# click.rich_click.STYLE_COMMANDS_TABLE_PAD_EDGE = True
#click.rich_click.STYLE_COMMANDS_TABLE_BOX = "DOUBLE"
click.rich_click.STYLE_COMMANDS_TABLE_BORDER_STYLE = "red"
click.rich_click.STYLE_COMMANDS_TABLE_ROW_STYLES = ["magenta", "yellow", "cyan", "green"]

console = Console()

# Configuration handling - use project root directory
PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "openarc-cli-config.yaml"

def save_cli_config(host: str, port: int):
    """Save server configuration to YAML config file."""
    config = {
        "server": {
            "host": host,
            "port": port
        },
        "created_by": "openarc-cli",
        "version": "1.0"
    }
    
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print(f"📝 [dim]Configuration saved to: {CONFIG_FILE}[/dim]")

def load_cli_config():
    """Load server configuration from YAML config file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
                if config and "server" in config:
                    return config["server"]
        except (yaml.YAMLError, FileNotFoundError, KeyError):
            console.print(f"[yellow]Warning: Could not read config file {CONFIG_FILE}[/yellow]")
    
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
        art.append("The CLI application   \n", style="white")
        console.print(art)
        return super().get_help(ctx)

@click.group(cls=ColoredAsciiArtGroup)
def cli():
    """
    Use this application to interface with the OpenArc server.
    
    Features:
    
    • Start the OpenArc server.
    
    • Load models into the OpenArc server.

    • Check the status of loaded models.

    • Unload models.

    • Query device properties.

    • Query installed devices.


    To get started add --help to one of the commands below to view its documentation.
    """

@cli.command()

@click.option('--model', 
              required=True, 
              help="""
              - Absolute path to model.

              - The dir name which stores the openvino model files is used in the API to identify the model.

              - The dir name is the same as the model name.
              """)

@click.option('--model-type', 
              type=click.Choice(['TEXT', 'VISION']),
              required=True, 
              default='TEXT',
              help="""

              - Type of model.

              """)

@click.option('--device', 
              required=True, 
              default='CPU', 
              help="""
              - Device: CPU, GPU.0, GPU.1, GPU.2, GPU.3, GPU.4, AUTO

              - GPU.0 is the first GPU, GPU.1 is the second GPU, etc.

              - AUTO will automatically select the best device.
              """)

@click.option('--use-cache/--no-use-cache',
              required=True, 
              default=True,
              help="""
              - Use cache for stateful models.

              - Edge cases may require disabling cache, probably based on model architecture.

              """)

@click.option('--dynamic-shapes/--no-dynamic-shapes',
              required=True, 
              default=True,
              help="""
              - Use dynamic shapes.
               
              - If false, the model will be loaded with static shapes.

              - OpenVINO IR usually use dynamic shapes but for NPU it must be disabled.

              """)

@click.option('--pad-token-id', 
              required=False, 
              type=int, 
              help="""
              - (pad)pad token ID

              - AutoTokenizers usually infers this from config.json but it's useful to configure explicitly.

              """
              )

@click.option('--eos-token-id', 
              required=False, 
              type=int, 
              help="""
                - (eos)end of sequence token id

                - AutoTokenizers usually infers this from config.json but it's useful to configure explicitly.  

                - When the eos token is set to the *incorrect* token the model will continue to generate tokens.
                
                - Pairing this with a target max_length is a good way to test performance.
                """
              
              )

@click.option('--bos-token-id', 
              required=False, 
              type=int, 
              help='beginning of sequence token ID')

@click.option('--num-streams', 
              required=False, 
              type=int, 
              default=None,
              show_default=True,
              help='Number of inference streams')

@click.option('--performance-hint', 
              required=False, 
              type=click.Choice(['LATENCY', 'THROUGHPUT', 'CUMULATIVE_THROUGHPUT']),
              default=None,
              show_default=True,
              help="""
              ---

              - High level performance hint.

              - Usually I use 'LATENCY' which locks to one CPU or one CPU socket.

              - It's best to use the documentation for this.

              https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html

              ---

              """
              )

@click.option('--inference-precision-hint', 
              required=False, 
              type=click.Choice(['fp32', 'f16', 'bf16', 'dynamic']),
              default=None,
              show_default=True,
              help="""
              ---

              - Controls precision during inference, at inference time.

                - Works on CPU and GPU.

              - Target device specific features.

              - Ex:'bf16' is probably best on CPUs which support AMX.
              
              """
              )

@click.option('--enable-hyper-threading', 
              required=False, 
              type=bool, 
              default=None,
              help="""
              ---

              - CPU ONLY --> Cannot be used with GPU.

              - Enable hyper-threading 

              - This is only relevant for Intel CPUs with hyperthreading i.e, two virtual cores per physical core.

              """
              )

@click.option('--inference-num-threads', 
              required=False, 
              type=int, 
              default=None,
              show_default=True,
              help="""
              ---

              - CPU ONLY --> Cannot be used with GPU.

              - Number of inference threads

              - More threads usually means faster inference. 

              - Therefore this can be used to constrain the number of threads used for inference.
              """
              )

@click.option('--scheduling-core-type', 
              required=False, 
              type=click.Choice(['ANY_CORE', 'PCORE_ONLY', 'ECORE_ONLY']),
              default=None,
              show_default=True,
              help="""
              ---

              - Advanced option to target p-cores or e-cores on CPUs which support it.

              - CPU ONLY --> Cannot be used with GPU.

              - [ANY_CORE]: Any core, so default for 'older' Intel CPUs. Default for most chips but no need to set.

              - [PCORE_ONLY]: Only run inference on threads which are performance cores.

              - [ECORE_ONLY]: Only run inference on threads which are efficency cores.
              ---
                """
              )

@click.pass_context
def load(ctx, model, model_type, device, use_cache, dynamic_shapes,
         pad_token_id, eos_token_id, bos_token_id, num_streams, performance_hint,
         inference_precision_hint, enable_hyper_threading, inference_num_threads,
         scheduling_core_type):
    """- Load a model."""
    cli_instance = OpenArcCLI()
    
    # Build load_config from arguments
    load_config = {
        "id_model": model,
        "architecture_type": model_type,
        "use_cache": use_cache,
        "device": device,
        "dynamic_shapes": dynamic_shapes,
    }
    
    # Add optional token IDs if provided
    if pad_token_id is not None:
        load_config["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        load_config["eos_token_id"] = eos_token_id
    if bos_token_id is not None:
        load_config["bos_token_id"] = bos_token_id
    
    # Build ov_config from arguments
    ov_config = {}
    if performance_hint is not None:
        ov_config["PERFORMANCE_HINT"] = performance_hint
    if inference_precision_hint is not None:
        ov_config["INFERENCE_PRECISION_HINT"] = inference_precision_hint
    if enable_hyper_threading is not None:
        ov_config["ENABLE_HYPER_THREADING"] = enable_hyper_threading
    if inference_num_threads not in (None, False):
        ov_config["INFERENCE_NUM_THREADS"] = inference_num_threads
    if scheduling_core_type is not None:
        ov_config["SCHEDULING_CORE_TYPE"] = scheduling_core_type
    if num_streams is not None:
        ov_config["NUM_STREAMS"] = num_streams
    
    # Prepare payload
    payload = {
        "load_config": load_config,
        "ov_config": ov_config if ov_config else {}
    }
    
    # Make API request
    url = f"{cli_instance.base_url}/optimum/model/load"
    
    try:
        console.print(f"🚀 [blue]Loading model:[/blue] {model}")
        response = requests.post(url, json=payload, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            console.print("✅ [green]Model loaded successfully![/green]")
        else:
            console.print(f"❌ [red]Error loading model: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.option('--model-id', required=True, help='Model ID to unload')
@click.pass_context
def unload(ctx, model_id):
    """
    - DELETE a model from memory. 
    """
    cli_instance = OpenArcCLI()

    # Make API request
    url = f"{cli_instance.base_url}/optimum/model/unload"
    params = {"model_id": model_id}
    
    try:
        console.print(f"🗑️  [blue]Unloading model:[/blue] {model_id}")
        response = requests.delete(url, params=params, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            result = response.json()
            console.print(f"✅ [green]{result['message']}[/green]")
        else:
            console.print(f"❌ [red]Error unloading model: {response.status_code}[/red]")
            console.print(f"[red]Response:[/red] {response.text}")
            ctx.exit(1)
            
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [red]Request failed:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """- GET Status of loaded models."""
    cli_instance = OpenArcCLI()
    
    url = f"{cli_instance.base_url}/optimum/status"
    
    try:
        console.print("📊 [blue]Getting model status...[/blue]")
        response = requests.get(url, headers=cli_instance.get_headers())
        
        if response.status_code == 200:
            result = response.json()
            loaded_models = result.get("loaded_models", {})
            total_models = result.get("total_models_loaded", 0)
            
            if not loaded_models:
                console.print("[yellow]No models currently loaded.[/yellow]")
            else:
                for model_id, model_info in loaded_models.items():
                    device = model_info.get("device", "unknown")
                    status_val = model_info.get("status", "unknown")
                    metadata = model_info.get("model_metadata", {})
                    model_type = metadata.get("architecture_type", "unknown")
                    use_cache = str(metadata.get("use_cache", "-"))
                    dynamic_shapes = str(metadata.get("dynamic_shapes", "-"))
                    pad_token_id = str(metadata.get("pad_token_id", "-"))
                    eos_token_id = str(metadata.get("eos_token_id", "-"))
                    bos_token_id = str(metadata.get("bos_token_id", "-"))
                    num_streams = str(metadata.get("NUM_STREAMS", "-"))
                    precision = str(metadata.get("INFERENCE_PRECISION_HINT", "-"))
                    perf_hint = str(metadata.get("PERFORMANCE_HINT", "-"))
                    inf_num_threads = str(metadata.get("INFERENCE_NUM_THREADS", "-"))
                    enable_ht = str(metadata.get("ENABLE_HYPER_THREADING", "-"))
                    sched_core_type = str(metadata.get("SCHEDULING_CORE_TYPE", "-"))

                    model_table = Table(show_header=False, box=None, pad_edge=False)

                    model_table.add_row(Text("Model Info", style="bold underline cyan"), "")
                    model_table.add_row("Model ID", f"[cyan]{model_id}[/cyan]")
                    model_table.add_row("Type", f"[yellow]{model_type}[/yellow]")
                    model_table.add_row("Status", f"[green]{status_val}[/green]" if status_val == "loaded" else f"[red]{status_val}[/red]")
                    model_table.add_row("", "")

                    # Device Info Section
                    model_table.add_row(Text("Device Info", style="bold underline magenta"), "")
                    model_table.add_row("Device", f"[magenta]{device}[/magenta]")
                    model_table.add_row("Use Cache", use_cache)
                    model_table.add_row("Dynamic Shapes", dynamic_shapes)
                    model_table.add_row("", "")

                    # Token IDs Section
                    model_table.add_row(Text("Token IDs", style="bold underline yellow"), "")
                    model_table.add_row("Pad Token ID", pad_token_id)
                    model_table.add_row("EOS Token ID", eos_token_id)
                    model_table.add_row("BOS Token ID", bos_token_id)
                    model_table.add_row("", "")

                    # Performance Settings Section
                    model_table.add_row(Text("Performance Settings", style="bold underline green"), "")
                    model_table.add_row("NUM_STREAMS", num_streams)
                    model_table.add_row("INFERENCE_PRECISION_HINT", precision)
                    model_table.add_row("PERFORMANCE_HINT", perf_hint)
                    model_table.add_row("INFERENCE_NUM_THREADS", inf_num_threads)
                    model_table.add_row("ENABLE_HYPER_THREADING", enable_ht)
                    model_table.add_row("SCHEDULING_CORE_TYPE", sched_core_type)

                    panel = Panel(
                        model_table,
                        title=f"🧩 Model: [bold]{model_id}[/bold]",
                        border_style="blue" if status_val == "loaded" else "red"
                    )
                    console.print(panel)
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

@tool.command('device-properties')
@click.pass_context
def device_properties(ctx):
    """
    - Query device properties for all devices.
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
        
        table = Table(title="📋 Available Devices")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Device", style="green")
        
        if not available_devices:
            console.print("❌ [red]No OpenVINO devices found![/red]")
            ctx.exit(1)
        
        for i, device in enumerate(available_devices, 1):
            table.add_row(str(i), device)
        
        console.print(table)
        console.print(f"\n✅ [green]OpenVINO runtime found {len(available_devices)} device(s)[/green]")
            
    except Exception as e:
        console.print(f"❌ [red]Error during device diagnosis:[/red] {e}")
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
    - 'start' reads --host and --openarc-port and saves them to the config file. Then it starts the server and will read
    """
    # Save server configuration for other CLI commands to use
    save_cli_config(host, openarc_port)
    
    console.print(f"🚀 [green]Starting OpenArc server on {host}:{openarc_port}[/green]")
    start_server(host=host, openarc_port=openarc_port)


if __name__ == "__main__":
    cli()



