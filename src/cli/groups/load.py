"""
Load command - Load one or more models from saved configuration.
"""
import click
import requests

from ..main import OpenArcCLI, cli, console
from ..utils import validate_model_path


@cli.command()
@click.argument('model_names', nargs=-1, required=True)
@click.pass_context
def load(ctx, model_names):
    """- Load one or more models from saved configuration.
    
    Examples:
        openarc load model_name
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
