#!/usr/bin/env python3
"""
OpenArc CLI Tool - Main entry point and core infrastructure.
"""
import os

import rich_click as click
from rich.console import Console
from rich.text import Text

click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
click.rich_click.STYLE_COMMANDS_TABLE_BORDER_STYLE = "red"
click.rich_click.STYLE_COMMANDS_TABLE_ROW_STYLES = ["magenta", "yellow", "cyan", "green"]

console = Console()


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
            from .modules.server_config import ServerConfig
            self._server_config = ServerConfig()
        return self._server_config
    
    @property
    def benchmark_db(self):
        """Lazy-load BenchmarkDB only when needed."""
        if self._benchmark_db is None:
            from .modules.benchmark import BenchmarkDB
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
                (" You know who ELSE uses OpenArc?\n", "white"),
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


# Import command groups to register them with the CLI
from .groups import add, bench, list, load, serve, status, tool, unload  # noqa: E402, F401


if __name__ == "__main__":
    cli()
