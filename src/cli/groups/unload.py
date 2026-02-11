"""
Unload command - Unload one or more models from registry and memory.
"""
import click
import requests

from ..main import OpenArcCLI, cli, console


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
