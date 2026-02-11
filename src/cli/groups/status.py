"""
Status command - GET Status of loaded models.
"""
import click
import requests
from rich.table import Table

from ..main import OpenArcCLI, cli, console


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
