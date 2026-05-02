#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import openvino as ov


EXCLUDED_NAME_PARTS = ("tokenizer", "detokenizer", "detokenize")


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


def inspect_model(core: ov.Core, xml_path: Path) -> Counter[str]:
    model = core.read_model(str(xml_path))
    counts: Counter[str] = Counter(op.get_type_name() for op in model.get_ops())

    print(f"\n=== {xml_path.name} ===")
    print(f"Friendly name: {model.get_friendly_name()}")
    print(f"Total ops: {sum(counts.values())}")
    print("Discovered ops:")

    for op_type, count in sorted(counts.items()):
        print(f"  {op_type}: {count}")

    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print discovered OpenVINO IR op types and counts for each non-tokenizer IR under one path.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="OpenVINO .xml file or directory containing .xml IR files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    xml_models = find_xml_models(args.path)

    if not xml_models:
        skipped = ", ".join(EXCLUDED_NAME_PARTS)
        raise RuntimeError(f"No non-tokenizer .xml OpenVINO IR files found under: {args.path} (skipped: {skipped})")

    core = ov.Core()

    print("OpenVINO IR Ops. These are nodes in the models graph.")
    print("OpenVINO version:", ov.__version__)
    print("Found IR files:")
    for xml in xml_models:
        print(f"  - {xml.name}")

    for xml in xml_models:
        inspect_model(core, xml)


if __name__ == "__main__":
    main()
