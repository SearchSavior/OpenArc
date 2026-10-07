"""
Serve command group - Start the OpenArc server.
"""
import os

import click

from src.server.utils.context import check_context_window_exceeded

from ..main import cli, console


@cli.group()
def serve():
    """
    - Start the OpenArc server.
    """
    pass


def _startup_context_window_warning(server_config, name: str):
    """Mirror the server-side check up-front so a pinned context_window larger
    than the real max_position_embeddings is warned about in the foreground.
    Advertisement-only; returns None (unset / "auto" / a config that can't be
    resolved) when there is nothing to warn about."""
    try:
        load_config = server_config.get_model_load_config(name)
    except Exception:
        return None
    if load_config is None:
        return None
    return check_context_window_exceeded(
        load_config.model_name,
        load_config.model_path,
        load_config.context_window,
    )


@serve.command("start")
@click.option("--host", type=str, default="0.0.0.0", show_default=True,
              help="""
              - Host to bind the server to
              """)
@click.option("--port",
              type=int,
              default=8000,
              show_default=True,
              help="""
              - Port to bind the server to
              """)
@click.option("--load-models", "--lm",
              required=False,
              help="Load models on startup. Specify once followed by space-separated model names.")
@click.option("--use-api-key", is_flag=True, default=False,
              help="Require OPENARC_API_KEY for all requests.")
@click.option("-v", "--verbose", count=True, default=0,
              help="Increase verbosity: -v warnings, -vv info + HTTP requests, -vvv debug, -vvvv debug incl. third-party libraries.")
@click.argument('startup_models', nargs=-1, required=False)
@click.pass_context
def start(ctx, host, port, load_models, use_api_key, verbose, startup_models):
    """
    - 'start' reads --host and --port from config or defaults to 0.0.0.0:8000

    Examples:
        openarc serve start
        openarc serve start --load-models model1 model2
        openarc serve start --lm Dolphin-X1 kokoro whisper
    """
    from ..modules.launch_server import start_server

    # config.yaml is never rewritten here: it is hand-authored and may carry
    # comments and ${VAR} references that a YAML round-trip would destroy.
    console.print(f"[dim]Using configuration: {ctx.obj.server_config.config_file}[/dim]")

    # Handle startup models
    models_to_load = []
    if load_models:
        models_to_load.append(load_models)
    if startup_models:
        models_to_load.extend(startup_models)

    if models_to_load:
        saved_model_names = ctx.obj.server_config.get_model_names()
        missing = [m for m in models_to_load if m not in saved_model_names]

        if missing:
            console.print("[yellow]Warning: Models not in config (will be skipped):[/yellow]")
            for m in missing:
                console.print(f"   • {m}")
            console.print("[dim]Use 'openarc list' to see saved configurations.[/dim]\n")

        os.environ["OPENARC_STARTUP_MODELS"] = ",".join(models_to_load)
        console.print(f"[blue]Models to load on startup:[/blue] {', '.join(models_to_load)}\n")

        # Warn up-front (foreground) for startup models that pin an over-large
        # context_window; register_load logs the same warning on the server side.
        # Models missing from the config are already flagged above and skipped.
        for name in models_to_load:
            if name in missing:
                continue
            warning = _startup_context_window_warning(ctx.obj.server_config, name)
            if warning:
                console.print(f"[red bold]{warning}[/red bold]\n")

    if use_api_key:
        if not os.getenv("OPENARC_API_KEY"):
            console.print("[red]Error: You chose to require an API key but OPENARC_API_KEY has not been set.[/red]")
            raise SystemExit(1)
        os.environ["OPENARC_API_KEY_REQUIRED"] = "true"
        console.print("[blue]OPENARC_API_KEY_REQUIRED=[/blue][green]True[/green] [dim][Clients connecting to the server must authenticate with OPENARC_API_KEY][/dim]")
    else:
        os.environ["OPENARC_API_KEY_REQUIRED"] = "false"
        console.print("[blue]OPENARC_API_KEY_REQUIRED=[/blue][yellow]False[/yellow] [dim][Clients do not need to authenticate.][/dim]")

    console.print(f"[green]Starting OpenArc server on {host}:{port}[/green]")
    start_server(host=host, port=port, verbose=verbose)
