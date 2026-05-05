#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import openvino as ov
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


EXCLUDED_NAME_PARTS = ("tokenizer", "detokenizer")
console = Console()


def should_skip_xml(path: Path) -> bool:
    name = path.name.lower()
    return any(part in name for part in EXCLUDED_NAME_PARTS)


def find_xml_models(path: Path) -> list[Path]:
    path = path.expanduser()

    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"Expected an OpenVINO .xml file, got: {path}")
        return [] if should_skip_xml(path) else [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return [xml for xml in sorted(path.rglob("*.xml")) if not should_skip_xml(xml)]


def format_bytes(num_bytes: int) -> str:
    mb = num_bytes / (1024.0 * 1024.0)
    if mb > 1000.0:
        gb = mb / 1024.0
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def get_node_opset(node: object) -> str:
    try:
        type_info = node.get_type_info()
        for attr in ("version_id", "version", "name"):
            value = getattr(type_info, attr, None)
            if value:
                raw = str(value)
                match = re.search(r"opset\s*([0-9]+)", raw, flags=re.IGNORECASE)
                if match:
                    return match.group(1)
                if raw.isdigit():
                    return raw

        raw_type_info = str(type_info)
        fallback_match = re.search(r"opset\s*([0-9]+)", raw_type_info, flags=re.IGNORECASE)
        if fallback_match:
            return fallback_match.group(1)
    except Exception:
        pass
    return "unknown"


def inspect_model(core: ov.Core, xml_path: Path) -> Counter[str]:
    model = core.read_model(str(xml_path))
    counts: Counter[str] = Counter(op.get_type_name() for op in model.get_ops())
    opset_counts: Counter[tuple[str, str]] = Counter(
        (get_node_opset(op), op.get_type_name()) for op in model.get_ops()
    )
    xml_size = xml_path.stat().st_size

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_row("Friendly name", f"[cyan]{model.get_friendly_name()}[/cyan]")
    summary.add_row("IR file size", f"[yellow]{format_bytes(xml_size)}[/yellow]")
    summary.add_row("Total ops", f"[green]{sum(counts.values())}[/green]")

    console.print()
    console.print(Panel(summary, title=f"IR: {xml_path.name}", border_style="blue", expand=False))

    ops_table = Table(show_header=True, header_style="bold magenta")
    ops_table.add_column("Opset", style="cyan")
    ops_table.add_column("Op Type", style="white")
    ops_table.add_column("Count", justify="center", style="white")

    for (opset, op_type), count in sorted(opset_counts.items()):
        ops_table.add_row(opset, op_type, str(count))

    console.print(ops_table)

    return counts


def parse_search_targets(search_targets: str | None) -> list[str]:
    if not search_targets:
        return []

    parsed = [target.strip() for target in search_targets.split(",") if target.strip()]
    if not parsed:
        raise ValueError("Search targets were provided, but none were valid after parsing.")
    return parsed


def print_target_counts(counts: Counter[str], targets: Iterable[str]) -> None:
    normalized = {op_name.lower(): count for op_name, count in counts.items()}
    target_table = Table(show_header=True, header_style="bold yellow")
    target_table.add_column("Target", style="white")
    target_table.add_column("Count", justify="center", style="white")
    target_table.add_column("Status", style="cyan")

    for target in targets:
        count = normalized.get(target.lower(), 0)
        status = "FOUND" if count > 0 else "MISSING"
        target_table.add_row(target, str(count), status)

    console.print(target_table)


def run_inspection(path: Path, search_targets: list[str] | None = None) -> None:
    xml_models = find_xml_models(path)

    if not xml_models:
        skipped = ", ".join(EXCLUDED_NAME_PARTS)
        raise RuntimeError(f"No non-tokenizer .xml OpenVINO IR files found under: {path} (skipped: {skipped})")

    core = ov.Core()

    console.print("[bold blue]OpenVINO IR Ops[/bold blue] [dim](model graph nodes)[/dim]")
    console.print(f"[white]OpenVINO version:[/white] [cyan]{ov.__version__}[/cyan]")

    files_table = Table(show_header=True, header_style="bold blue")
    files_table.add_column("IR Files Found", style="white")
    for xml in xml_models:
        files_table.add_row(xml.name)
    console.print(files_table)

    aggregate_counts: Counter[str] = Counter()
    for xml in xml_models:
        model_counts = inspect_model(core, xml)
        aggregate_counts.update(model_counts)
        if search_targets:
            print_target_counts(model_counts, search_targets)

    if search_targets:
        console.print("\n[bold yellow]Aggregate Target Summary (All Inspected IRs)[/bold yellow]")
        print_target_counts(aggregate_counts, search_targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print discovered OpenVINO IR op types and counts for each non-tokenizer IR under one path.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="OpenVINO .xml file or directory containing .xml IR files.",
    )
    parser.add_argument(
        "-s",
        "--search-targets",
        default=None,
        help="Comma delimited list of ops you want to check for.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    search_targets = parse_search_targets(args.search_targets)
    run_inspection(args.path, search_targets=search_targets)


if __name__ == "__main__":
    main()
