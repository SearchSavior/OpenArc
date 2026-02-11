"""
Bench command - Benchmark inference with pseudo-random input tokens.
"""
import uuid
from pathlib import Path

import click
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..main import cli, console
from ..utils import validate_model_path


@cli.command()
@click.argument('model_name')
@click.option('--input_tokens', '--p', multiple=True, default=['512'],
              help='Number of prompt tokens. Can be comma-separated (e.g., --p 16,32) or specified multiple times (e.g., -p 16 -p 32). Default: 512')
@click.option('--max_tokens', '--n', multiple=True, default=['128'],
              help='Number of tokens to generate. Can be comma-separated or specified multiple times. Default: 128')
@click.option('--runs', '--r', default=5, type=int,
              help='Number of times to repeat each benchmark. Default: 5')
@click.option('--temperature', '--temp', default=None, type=float,
              help='Sampling temperature (default: 1.0)')
@click.option('--top-k', '--k', default=None, type=int,
              help='Top-k sampling (default: 50)')
@click.option('--top-p', '--p-nucleus', default=None, type=float,
              help='Top-p nucleus sampling (default: 1.0)')
@click.option('--repetition-penalty', '--rep', default=None, type=float,
              help='Repetition penalty (default: 1.0)')
@click.pass_context
def bench(ctx, model_name, input_tokens, max_tokens, runs, temperature, top_k, top_p, repetition_penalty):
    """- Benchmark inference with pseudo-random input tokens.
    
    Examples:
        openarc bench Dolphin-X1
        openarc bench Dolphin-X1 --p 512 --n 128 -r 10
        openarc bench Dolphin-X1 --p 16,32,64 --n 128,256
        openarc bench Dolphin-X1 -p 16 -p 32 -n 128 -n 256
    """
    from ..modules.benchmark import OpenArcBenchmarks
    from ..main import OpenArcCLI
    
    cli_instance = OpenArcCLI(server_config=ctx.obj.server_config)
    
    # Parse input_tokens and max_tokens (handle comma-separated and multiple invocations)
    p_values = []
    for pt in input_tokens:
        p_values.extend([int(x.strip()) for x in pt.split(',')])
    
    n_values = []
    for nt in max_tokens:
        n_values.extend([int(x.strip()) for x in nt.split(',')])
    
    # Check if model exists
    try:
        console.print("[cyan]working...[/cyan]\n")
        models_url = f"{cli_instance.base_url}/v1/models"
        models_response = requests.get(models_url, headers=cli_instance.get_headers())
        
        if models_response.status_code != 200:
            console.print(f"[red]Failed to get model list: {models_response.status_code}[/red]")
            ctx.exit(1)
        
        models_data = models_response.json()
        available_models = [m['id'] for m in models_data.get('data', [])]
        
        if model_name not in available_models:
            console.print(f"[red]'{model_name}' not found in loaded models[/red]")
            console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
            console.print("[dim]Use 'openarc status' to see loaded models.[/dim]")
            ctx.exit(1)
        
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Request failed:[/red] {e}")
        ctx.exit(1)
    
    # Get model path from config to generate input tokens
    model_config = ctx.obj.server_config.get_model_config(model_name)
    if not model_config:
        console.print(f"[red]Model configuration not found for '{model_name}'[/red]")
        console.print("[yellow]Cannot generate random tokens without model path.[/yellow]")
        console.print("[blue]Use 'openarc list' to see saved configurations.[/blue]")
        ctx.exit(1)
    
    model_path = model_config.get('model_path')
    if not model_path:
        console.print("[red]model_path not found in configuration[/red]")
        ctx.exit(1)
    
    # Validate model path
    if not validate_model_path(model_path):
        console.print(f"[red]Model file check failed! {model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
        ctx.exit(1)
    
    # Run benchmarks
    console.print(f"input tokens: {p_values}")
    console.print(f"max tokens:   {n_values}")
    console.print(f"runs: {runs}\n")
    
    # Generate unique run_id for this benchmark session
    run_id = str(uuid.uuid4())
    
    total_runs = len(p_values) * len(n_values) * runs
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running... (0/{total_runs})", total=total_runs)
        
        run_count = 0
        for p in p_values:
            for n in n_values:
                for r in range(runs):
                    run_count += 1
                    progress.update(task, description=f"[dim]benching...[/dim] ({run_count}/{total_runs}) [p={p}, n={n}, r={r+1}/{runs}]")
                    
                    try:
                        # Generate random input tokens
                        input_ids = OpenArcBenchmarks.random_input_ids(model_path, p)
                        
                        # Make benchmark request
                        bench_url = f"{cli_instance.base_url}/openarc/bench"
                        payload = {
                            "model": model_name,
                            "input_ids": input_ids,
                            "max_tokens": n
                        }

                        # Add optional parameters if provided
                        if temperature is not None:
                            payload["temperature"] = temperature
                        if top_k is not None:
                            payload["top_k"] = top_k
                        if top_p is not None:
                            payload["top_p"] = top_p
                        if repetition_penalty is not None:
                            payload["repetition_penalty"] = repetition_penalty

                        bench_response = requests.post(
                            bench_url,
                            headers=cli_instance.get_headers(),
                            json=payload
                        )
                        
                        if bench_response.status_code != 200:
                            console.print(f"\n[red]Benchmark request failed: {bench_response.status_code}[/red]")
                            console.print(f"[red]Response:[/red] {bench_response.text}")
                            continue
                        
                        metrics = bench_response.json().get('metrics', {})
                        
                        # Store individual result
                        result = {
                            'p': p,
                            'n': n,
                            'run': r + 1,
                            'ttft': metrics.get('ttft (s)', 0),
                            'tpot': metrics.get('tpot (ms)', 0),
                            'prefill_throughput': metrics.get('prefill_throughput (tokens/s)', 0),
                            'decode_throughput': metrics.get('decode_throughput (tokens/s)', 0),
                            'decode_duration': metrics.get('decode_duration (s)', 0),
                            'input_token': metrics.get('input_token', 0),
                            'new_token': metrics.get('new_token', 0),
                            'total_token': metrics.get('total_token', 0),
                        }
                        results.append(result)
                        
                        # Save result to database
                        ctx.obj.benchmark_db.save_result(model_name, result, run_id)
                        
                    except Exception as e:
                        console.print(f"\n[yellow]Error in run {r+1}: {e}[/yellow]")
                        continue
                    
                    progress.advance(task)
    
    # Display results
    console.print("\n")
    
    if not results:
        console.print("[red]No benchmark results collected![/red]")
        ctx.exit(1)
    
    
    
    
    model_path_name = Path(model_path).name
    console.print(f"\n[blue]{model_path_name}[/blue]\n")

    # Create results table with visible lines
    results_table = Table(show_header=True, header_style="bold")
    results_table.add_column("[cyan]run[/cyan]", justify="right")
    results_table.add_column("[cyan]p[/cyan]", justify="right")
    results_table.add_column("[cyan]n[/cyan]", justify="right")
    results_table.add_column("[cyan]ttft(s)[/cyan]", justify="right")
    results_table.add_column("[cyan]tpot(ms)[/cyan]", justify="right")
    results_table.add_column("[cyan]prefill(t/s)[/cyan]", justify="right")
    results_table.add_column("[cyan]decode(t/s)[/cyan]", justify="right")
    results_table.add_column("[cyan]duration(s)[/cyan]", justify="right")

    
    for result in results:
        results_table.add_row(
            str(result['run']),
            str(result['p']),
            str(result['n']),
            f"{result['ttft']:.2f}",
            f"{result['tpot']:.2f}",
            f"{result['prefill_throughput']:.1f}",
            f"{result['decode_throughput']:.1f}",
            f"{result['decode_duration']:.2f}"
            )
    
    console.print(results_table)
    

    console.print(f"[dim]Total: {len(results)} runs[/dim]")
