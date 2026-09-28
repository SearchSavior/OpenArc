"""Generate a small, real OpenVINO GenAI LLM model for the worker tests.

This produces ``tests/integration/fixtures/tiny-gpt2-ov`` -- a ~4 MB convert of
``sshleifer/tiny-gpt2`` that ``openvino_genai.LLMPipeline`` can *actually load
and run on the CPU plugin*. It exists so the integration test that verifies a
respawn after an unrecoverable OpenVINO error needs no GPU, no 0.6 B model, and
no pre-existing files under ``TEST_MODEL_PATH`` -- just this tiny artifact on
disk (or, if it is absent, a fresh download+convert via ``os.system``).

It is intentionally a *separate* script, run with the ``optimum-intel`` venv,
because converting a HuggingFace model to OpenVINO IR pulls in ``optimum`` /
``optimum-intel`` -- which lives in its own venv (``pip install "optimum-intel"``,
see ``setup.sh``), not in the ``openarc-devel`` venv that runs the server and
the tests. It therefore takes no OpenArc dependencies at all.

Run it once with::

    ~/.local/pyvenv/optimum-intel/bin/python tests/integration/_generate_tiny_ovllm.py
    # or into a custom dir:
    OPTIMUM_INTEL .../python tests/integration/_generate_tiny_ovllm.py --out-dir /tmp/m

``--out-dir`` defaults to ``<this dir>/fixtures/tiny-gpt2-ov`` so a plain run
keeps the artifact inside the repo where it can be committed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from optimum.intel import OVModelForCausalLM

# The smallest causal LM that (a) optimum-intel converts cleanly and (b)
# openvino_genai loads + runs on CPU in ~0.1 s. The resulting artifact is a few
# MB and small enough to commit (see the module docstring).
DEFAULT_MODEL = "sshleifer/tiny-gpt2"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "fixtures" / "tiny-gpt2-ov"

# Files that make a directory a loadable openvino_genai.LLMPipeline input.
_MODEL_FILES = (
    "openvino_model.xml",
    "openvino_model.bin",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _copy_model_files(staging: Path, out_dir: Path) -> None:
    """Keep only the IR + tokenizer files in out_dir (drop weights/hubs)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _MODEL_FILES:
        src = staging / name
        if src.exists():
            shutil.copy(src, out_dir / name)


def generate(out_dir: Path, model: str = DEFAULT_MODEL) -> None:
    cache_dir = Path(os.environ.get("OPENARC_TEST_MODEL_CACHE", "/tmp/openarc-test-models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = snapshot_download(model, cache_dir=cache_dir)
    print(f"downloaded {model} -> {hf_dir}")

    # export=True compiles the Python model to OpenVINO IR while instantiating;
    # save_pretrained then writes the IR + tokenizer + config to `staging`.
    staging = cache_dir / f"{Path(hf_dir).name}-ov-out"
    if staging.exists():
        shutil.rmtree(staging)
    pipeline = OVModelForCausalLM.from_pretrained(hf_dir, export=True, load_in_8bit=False)
    pipeline.save_pretrained(str(staging))

    import transformers

    transformers.AutoTokenizer.from_pretrained(hf_dir).save_pretrained(str(staging))
    _copy_model_files(staging, out_dir)
    total = sum(
        p.stat().st_size
        for p in out_dir.iterdir()
        if p.is_file()
    )
    print(f"wrote {sorted(p.name for p in out_dir.iterdir())} total={total} bytes ({total/1e6:.2f} MB)")
    if total > 10 * 1024 * 1024:
        print("WARNING: artifact exceeds 10 MiB; consider not committing it.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    generate(Path(args.out_dir).expanduser(), args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
