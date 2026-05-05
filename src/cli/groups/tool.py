"""
Tool command group - Utility scripts.
"""
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from ..main import cli, console


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
        from ..modules.device_query import DeviceDataQuery
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
        from ..modules.device_query import DeviceDiagnosticQuery
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


@tool.command("inspect-ir")
@click.argument("model_ref", required=True)
@click.option(
    "-s",
    "--search-targets",
    required=False,
    default=None,
    help="Comma delimited list of ops you want to check for.",
)
@click.pass_context
def inspect_ir(ctx, model_ref, search_targets):
    """
    - Inspect OpenVINO IR ops by model name or model path value.
    """
    resolved_path = None
    model_config = ctx.obj.server_config.get_model_config(model_ref)

    if model_config:
        resolved_path = model_config.get("model_path")
        if not resolved_path:
            console.print(f"[red]No model_path found in configuration for:[/red] {model_ref}")
            ctx.exit(1)
    else:
        candidate_path = Path(model_ref).expanduser()
        if not candidate_path.exists():
            console.print(
                f"[red]'{model_ref}' is not a saved model name and does not exist as a path.[/red]"
            )
            console.print("[yellow]Tip: Use 'openarc list' to see saved configurations.[/yellow]")
            ctx.exit(1)

        has_model_xml = False
        if candidate_path.is_file():
            has_model_xml = candidate_path.name.lower().endswith("_model.xml")
        elif candidate_path.is_dir():
            has_model_xml = any(
                p.is_file() and p.name.lower().endswith("_model.xml")
                for p in candidate_path.rglob("*")
            )

        if not has_model_xml:
            console.print(
                f"[red]Path must contain at least one '*_model.xml' file:[/red] {candidate_path}"
            )
            ctx.exit(1)

        resolved_path = candidate_path

    try:
        from ..modules.inspect_openvino_ir import parse_search_targets, run_inspection

        targets = parse_search_targets(search_targets)
        run_inspection(Path(resolved_path), search_targets=targets)
    except Exception as e:
        console.print(f"[red]Error inspecting OpenVINO IR:[/red] {e}")
        ctx.exit(1)
