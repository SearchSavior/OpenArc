"""
Serve command group - Start the OpenArc server.
"""
import os

import click

from ..main import cli, console


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
@click.option("--force-recompile", "--fr", "force_recompile",
              is_flag=True, default=False,
              help="Force a full recompile at startup for every model loaded with "
              "--load-models/--lm: invalidate each model's compiled-model cache and "
              "rebuild from the IR even if its config is unchanged (no OpenVINO cache "
              "reuse). Use after swapping model files/host, or just to be sure each "
              "model was freshly compiled.")
@click.option("-v", "--verbose", count=True, default=0,
              help="Increase verbosity: -v warnings, -vv info + HTTP requests, -vvv debug, -vvvv debug incl. third-party libraries.")
@click.argument('startup_models', nargs=-1, required=False)
@click.pass_context
def start(ctx, host, port, load_models, use_api_key, force_recompile, verbose, startup_models):
    """
    - 'start' reads --host and --port from config or defaults to 0.0.0.0:8000

    Examples:
        openarc serve start
        openarc serve start --load-models model1 model2
        openarc serve start --lm Dolphin-X1 kokoro whisper
    """
    from ..modules.launch_server import start_server

    # Save server configuration for other CLI commands to use
    config_path = ctx.obj.server_config.save_server_config(host, port)
    console.print(f"[dim]Configuration saved to: {config_path}[/dim]")

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

    if use_api_key:
        if not os.getenv("OPENARC_API_KEY"):
            console.print("[red]Error: You chose to require an API key but OPENARC_API_KEY has not been set.[/red]")
            raise SystemExit(1)
        os.environ["OPENARC_API_KEY_REQUIRED"] = "true"
        console.print("[blue]OPENARC_API_KEY_REQUIRED=[/blue][green]True[/green] [dim][Clients connecting to the server must authenticate with OPENARC_API_KEY][/dim]")
    else:
        os.environ["OPENARC_API_KEY_REQUIRED"] = "false"
        console.print("[blue]OPENARC_API_KEY_REQUIRED=[/blue][yellow]False[/yellow] [dim][Clients do not need to authenticate.][/dim]")

    # --force-recompile / --fr: when set, force every model loaded at startup to
    # recompile (invalidate its compiled-model cache and rebuild from the IR) even
    # if its config is unchanged. The server reads it via this env var in its
    # lifespan and forwards it to ModelRegistry.register_load(force_recompile=...).
    os.environ["OPENARC_FORCE_RECOMPILE"] = "true" if force_recompile else "false"
    if force_recompile:
        console.print("[blue]OPENARC_FORCE_RECOMPILE=[/blue][green]True[/green] [dim][Startup loads force a full recompile (no cache reuse)][/dim]")
    else:
        console.print("[blue]OPENARC_FORCE_RECOMPILE=[/blue][yellow]False[/yellow] [dim][Startup loads reuse the OpenVINO cache when the config is unchanged][/dim]")

    console.print(f"[green]Starting OpenArc server on {host}:{port}[/green]")
    start_server(host=host, port=port, verbose=verbose)
