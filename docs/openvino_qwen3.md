# Converting Qwen3-Reranker to OpenVINO IR for OpenArc

Working recipe for converting a Qwen3-Reranker HuggingFace checkpoint to INT8 OpenVINO IR, validated on `Qwen/Qwen3-Reranker-4B` (2026-04-24).

## Why not `optimum-cli`

The obvious approach — `optimum-cli export openvino --weight-format int8` — **silently truncated** the output on every combination we tried (optimum-intel 1.27.0.dev + openvino 2026.1.0.dev, plus the notebook-pinned transformers 4.55.4 / torch 2.9.1 / openvino 2026.1.0 release). NNCF reported 100% weight compression, the process exited 0 with no traceback, but it wrote a **0-byte `openvino_model.xml`** and a ~13 MB stub `openvino_model.bin`. Bypassing the HF cache and using a local source copy did not help.

The same stack via the `optimum.intel` Python API produced a correct 4.03 GB bin + 3.16 MB xml on the first attempt. Use the Python API.

## Prerequisites

OpenArc's `.venv` already has everything needed — `optimum[openvino]`, `openvino`, `nncf`, `transformers`, `torch`. No extra install.

Source model on a writable local path. If you rely on the HF Hub cache, make sure `~/.cache/huggingface/hub/models--<org>--<name>/` is writable by the current user. A root-owned cache (e.g. populated by a container) produces `Permission denied` warnings during `.no_exist/` writes that are *technically* non-fatal for load, but correlate with other breakage — `chown -R $USER:$USER` the cache dirs if you see them.

## Conversion

```python
# convert_qwen3_reranker.py
import os, shutil
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from openvino_tokenizers import convert_tokenizer
import openvino as ov
from transformers import AutoTokenizer

SRC = "Qwen/Qwen3-Reranker-4B"          # HF id or local path
OUT = "/data/openvino-models/reranker/qwen3-4b-reranker"

os.makedirs(OUT, exist_ok=True)

# 1. Load + convert + INT8 weight-only compression
qc = OVWeightQuantizationConfig(bits=8, sym=False)
model = OVModelForCausalLM.from_pretrained(
    SRC,
    export=True,
    use_cache=False,        # reranker does a single scoring forward pass; no KV cache
    quantization_config=qc,
    compile=False,
)
model.save_pretrained(OUT)  # writes openvino_model.xml/.bin + openvino_config.json

# 2. Copy HF tokenizer artifacts (save_pretrained does NOT)
tok = AutoTokenizer.from_pretrained(SRC, padding_side="left")
tok.save_pretrained(OUT)

# 3. Emit OpenVINO tokenizer + detokenizer (optional but standard)
ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
ov.save_model(ov_tok,   os.path.join(OUT, "openvino_tokenizer.xml"))
ov.save_model(ov_detok, os.path.join(OUT, "openvino_detokenizer.xml"))
```

Run with `PYTHONUNBUFFERED=1` so progress bars flush. Expected peak RSS is roughly `model_fp_size + NNCF_overhead`, so plan on ≥16 GB RAM + swap headroom for the 4B. The compression phase is CPU-bound and took ~20 s on this host; the forward-pass trace and save together add another minute or two.

Quantization notes:
- `OVWeightQuantizationConfig(bits=8, sym=False)` is weight-only INT8 asymmetric, per-channel — the same thing `--weight-format int8` does with NNCF internally. No calibration dataset needed.
- Do **not** use `--quant-mode int8` / full activation quantization for a reranker without a domain-matched calibration set; accuracy drops quickly.
- For 4-bit, switch to `bits=4` and consider `sym=True`, `group_size=128`, and optionally `dataset="wikitext2"` with `awq=True, scale_estimation=True` for better recovery. INT8 is the right default for rerankers.

## Verification

Quick smoke test — load the exported model and run one forward pass:

```python
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

m = OVModelForCausalLM.from_pretrained(OUT, device="CPU", use_cache=False, export=False)
tok = AutoTokenizer.from_pretrained(OUT, padding_side="left")
inp = tok("Query: Paris\nDocument: Paris is the capital of France.\nRelevant:", return_tensors="pt")
with torch.no_grad():
    out = m(**inp)
print(out.logits.shape)   # expect (1, seq_len, 151669)
```

A vocab dimension of 151669 and a non-empty bin/xml on disk are the two signals that the export is real — not a stub.

## Wiring into OpenArc

Add the model to `openarc_config.json` alongside any existing reranker entry:

```json
"qwen3-4b-reranker": {
  "model_name": "qwen3-4b-reranker",
  "model_path": "/data/openvino-models/reranker/qwen3-4b-reranker",
  "model_type": "rerank",
  "engine": "optimum",
  "device": "GPU",
  "runtime_config": {},
  "vlm_type": null
}
```

`engine: optimum` wires it into `src/engine/optimum/optimum_rr.py`, which uses `AutoTokenizer` + an `OVModelForCausalLM` forward pass per (query, document) pair — so the HF tokenizer files in the output dir are what gets loaded at runtime. The `openvino_tokenizer.xml` produced above is unused by this engine today but is standard for OV IR packages.
