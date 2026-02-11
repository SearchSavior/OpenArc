"""
List command - List saved model configurations.
"""
import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..main import cli, console


@cli.command("list")
@click.argument('model_name', required=False)
@click.option('--v', 'verbose', is_flag=True, help='Show config for model_name.')
@click.option('--rm', 'remove', is_flag=True, help='Remove a model from config.')
@click.pass_context
def list(ctx, model_name, verbose, remove):
    """- List saved model configurations.
       
       - Remove a model configuration.
    
    Examples:
        openarc list                    # List all model names
        openarc list model_name --v     # Show metadata for specific model
        openarc list model_name --rm    # Remove a model configuration
    """
    if remove:
        if not model_name:
            console.print("[red]Error:[/red] model_name is required when using --remove")
            ctx.exit(1)
        
        # Check if model exists before trying to remove
        if not ctx.obj.server_config.model_exists(model_name):
            console.print(f"{model_name} [red]not found[/red]")
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
    
    # Case 1: Show metadata for specific model with -v flag
    if model_name and verbose:
        if model_name not in models:
            console.print(f"[red]Model not found:[/red] {model_name}")
            console.print("[yellow]Use 'openarc list' to see available configurations.[/yellow]")
            ctx.exit(1)
        
        model_config = models[model_name]
        
        # Create a table for the model configuration
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
        return
    
    # Case 2: Show only model names (default behavior)
    console.print(f"[blue]Saved Model Configurations ({len(models)}):[/blue]\n")
    
    for name in models.keys():
        console.print(f"  [cyan]{name}[/cyan]")
    
    console.print("[dim]Use 'openarc list <model_name> --v' to see model metadata.[/dim]")
    console.print("[dim]Use 'openarc list <model_name> --rm' to remove a configuration.[/dim]")
