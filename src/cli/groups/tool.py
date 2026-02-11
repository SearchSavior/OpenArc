"""
Tool command group - Utility scripts.
"""
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
