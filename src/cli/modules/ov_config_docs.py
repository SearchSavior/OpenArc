"""Dynamic help for the ``--runtime-config`` and ``--scheduler-config`` options.

The available keys are discovered at call time from the installed ``openvino`` /
``openvino_genai`` packages so the listing always matches the version actually
loaded. Where a package does not document a key, a curated description is used.
If the packages (or a GPU device) are unavailable, a curated fallback list is
used so the help still works on machines without a GPU or without OpenVINO
importable.

Invoke from the CLI with ``openarc add --runtime-config --help`` or
``openarc add --scheduler-config --help`` (the literal value ``--help`` is
treated as a sentinel, not as a JSON payload).
"""
from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Optional, Tuple

_WRAP_WIDTH = 100

# ---------------------------------------------------------------------------
# Curated descriptions for OpenVINO GPU plugin compile-config keys (used for
# keys the plugin does not document, and as the fallback list when no GPU /
# OpenVINO is available). Memory-relevant keys are documented first.
# ---------------------------------------------------------------------------
_RUNTIME_KEY_DOCS: Dict[str, str] = {
    "OFFLOAD_RATIO": (
        "Fraction (0.0-1.0) of the model offloaded to host memory. 0.0 = fully on "
        "GPU (default). Raise it to fit a model that is too large for the GPU, at a "
        "speed cost. A direct way to reduce GPU memory for an oversized model."
    ),
    "INFERENCE_PRECISION_HINT": (
        "Compute precision. Must be an OpenVINO element type short name (e.g. "
        "\"f16\", NOT \"float16\"). Common values: f16 (default), bf16, f32. Lower "
        "precision uses less activation memory."
    ),
    "KV_CACHE_PRECISION": (
        "Precision for KV-cache compression. Must be an OpenVINO element type "
        "short name (e.g. \"f16\", NOT \"float16\"). Values: f16 (full precision, "
        "default), bf16, f32, u8/i8 (8-bit, ~half the KV-cache memory), u4/i4 "
        "(4-bit, ~quarter the KV-cache memory, with an accuracy trade-off). Lower "
        "precision = less KV-cache memory."
    ),
    "NUM_STREAMS": (
        "Number of parallel inference streams (default 1). Keep 1 for maximum "
        "memory headroom; more streams increases peak GPU memory."
    ),
    "PERFORMANCE_HINT": (
        "Performance mode: LATENCY (default) / THROUGHPUT / BALANCED. THROUGHPUT "
        "allocates more memory; use LATENCY when memory is tight."
    ),
    "EXECUTION_MODE_HINT": (
        "Finer-grained execution-mode hint (successor to PERFORMANCE_HINT)."
    ),
    "GPU_ENABLE_LARGE_ALLOCATIONS": (
        "Enable large memory allocations (bool)."
    ),
    "ENABLE_CPU_PINNING": (
        "Pin host memory for transfers (bool); relevant when offloading."
    ),
    "ENABLE_CPU_RESERVATION": (
        "Reserve host memory (bool)."
    ),
    "COMPILATION_NUM_THREADS": (
        "Threads used during compilation (affects compile-time peak)."
    ),
    "PERFORMANCE_HINT_NUM_REQUESTS": (
        "Number of requests the PERFORMANCE_HINT targets."
    ),
    "CACHE_DIR": "Directory for the compiled-model cache.",
    "CACHE_MODE": "Compiled-model cache mode (NULL / SINGLE_FILE / MULTI_FILE).",
    "DEVICE_ID": "Index of the GPU device to use.",
    "GPU_ENABLE_SDPA_OPTIMIZATION": "Enable the SDPA attention optimization (bool).",
    "GPU_ENABLE_LORA_OPERATION": "Enable LoRA operation (bool).",
    "GPU_ENABLE_LOOP_UNROLLING": "Enable loop unrolling (bool).",
    "GPU_DISABLE_WINOGRAD_CONVOLUTION": "Disable Winograd convolution (bool).",
    "GPU_HOST_TASK_PRIORITY": "Host task priority.",
    "GPU_QUEUE_PRIORITY": "Queue priority.",
    "GPU_QUEUE_THROTTLE": "Queue throttle.",
    "MODEL_PRIORITY": "Model priority.",
    "PERF_COUNT": "Enable performance counters.",
    "CONFIG_FILE": "Path to a config file with additional plugin options.",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "Dynamic quantization group size.",
    "ACTIVATIONS_SCALE_FACTOR": "Activations scale factor.",
    "WEIGHTS_PATH": "Path to external weights.",
}
_RUNTIME_GENERIC_DOC = "OpenVINO GPU plugin compile-config key."


def _runtime_description(key: str) -> str:
    return _RUNTIME_KEY_DOCS.get(key, _RUNTIME_GENERIC_DOC)


# ---------------------------------------------------------------------------
# scheduler_config discovery (openvino_genai.SchedulerConfig)
# ---------------------------------------------------------------------------
def _parse_scheduler_doc(doc: str, field_names: List[str]) -> Dict[str, str]:
    """Parse the openvino_genai.SchedulerConfig docstring parameter sections into
    ``{field: one-line usage}``. Indent-based: a least-indented line that names a
    known field starts it; more-indented lines are continuations."""
    fn_set = set(field_names)
    lines = (doc or "").splitlines()
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return {}
    base = min(len(l) - len(l.lstrip(" ")) for l in nonempty)
    fields: Dict[str, str] = {}
    order: List[str] = []
    current: Optional[str] = None
    for l in lines:
        if not l.strip():
            continue
        indent = len(l) - len(l.lstrip(" "))
        text = l.strip()
        if indent > base:
            if current is not None:
                fields[current] = (fields[current] + " " + text).strip()
            continue
        # indent == base: a field start, a section header, or the title.
        if text.endswith(":") and text[:-1].strip() not in fn_set:
            current = None  # section header / title-with-colon
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(:)?\s*(.*)$", text)
        if m and m.group(1) in fn_set:
            current = m.group(1)
            fields[current] = (m.group(3) or "").strip()
            order.append(current)
        else:
            current = None
    return {k: re.sub(r"\s+", " ", fields[k]).strip() for k in order}


def _scheduler_fields_from_package() -> Optional[Tuple[List[str], Dict[str, str], Dict[str, str]]]:
    """Return ``(scalar_fields, {field: description}, {field: default_str})`` from the
    live ``openvino_genai.SchedulerConfig``, or ``None`` if the package is
    unavailable. Scalar fields are the int/bool/str/float attributes (nested config
    structs and methods are excluded)."""
    try:
        from openvino_genai import SchedulerConfig

        sc = SchedulerConfig()
    except Exception:
        return None
    doc = SchedulerConfig.__doc__ or ""
    scalar: List[str] = []
    defaults: Dict[str, str] = {}
    for a in dir(sc):
        if a.startswith("_"):
            continue
        try:
            val = getattr(sc, a)
        except Exception:
            continue
        if isinstance(val, bool):
            scalar.append(a)
            defaults[a] = "true" if val else "false"
        elif isinstance(val, (int, float, str)):
            scalar.append(a)
            defaults[a] = str(val)
    descs = _parse_scheduler_doc(doc, scalar)
    return scalar, descs, defaults


def _scheduler_schema_fields() -> Tuple[List[str], Dict[str, str]]:
    """Fallback: field names + descriptions from OpenArc's own SchedulerConfigSchema."""
    try:
        from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import (
            SchedulerConfigSchema,
        )

        fields = SchedulerConfigSchema.model_fields
        return list(fields.keys()), {k: (v.description or "").strip() for k, v in fields.items()}
    except Exception:
        return [], {}


# ---------------------------------------------------------------------------
# runtime_config discovery (OpenVINO GPU plugin)
# ---------------------------------------------------------------------------
def _runtime_keys_from_package() -> Optional[List[str]]:
    """Return the settable (RW) OpenVINO GPU plugin compile-config keys, or ``None``
    if no GPU device / OpenVINO is available."""
    try:
        import openvino as ov

        core = ov.Core()
    except Exception:
        return None
    dev = next((d for d in core.available_devices if d.startswith("GPU")), None)
    if not dev:
        return None
    try:
        props = core.get_property(dev, "SUPPORTED_PROPERTIES")
    except Exception:
        return None
    if not isinstance(props, dict):
        return None
    return sorted(k for k, v in props.items() if v == "RW")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _render(rows: List[Tuple[str, str, str]]) -> str:
    """Render ``(key, default, description)`` rows as a readable block."""
    if not rows:
        return "  (no keys could be discovered)"
    out: List[str] = []
    for key, default, desc in rows:
        head = f"  {key}"
        if default:
            head += f"   [default: {default}]"
        out.append(head)
        if desc:
            out.append(
                textwrap.fill(
                    desc,
                    width=_WRAP_WIDTH,
                    initial_indent="      ",
                    subsequent_indent="      ",
                )
            )
    return "\n".join(out)


# Keys shown first (the direct GPU-memory levers); the rest follow alphabetically.
_SCHEDULER_PRIORITY = [
    "cache_size",
    "num_kv_blocks",
    "max_num_seqs",
    "max_num_batched_tokens",
    "num_linear_attention_blocks",
    "cache_interval_multiplier",
    "enable_prefix_caching",
    "use_cache_eviction",
    "use_sparse_attention",
    "dynamic_split_fuse",
]
_RUNTIME_PRIORITY = [
    "OFFLOAD_RATIO",
    "KV_CACHE_PRECISION",
    "INFERENCE_PRECISION_HINT",
    "NUM_STREAMS",
    "PERFORMANCE_HINT",
    "EXECUTION_MODE_HINT",
    "GPU_ENABLE_LARGE_ALLOCATIONS",
    "ENABLE_CPU_PINNING",
    "ENABLE_CPU_RESERVATION",
    "COMPILATION_NUM_THREADS",
    "PERFORMANCE_HINT_NUM_REQUESTS",
    "CACHE_MODE",
    "CACHE_DIR",
    "DEVICE_ID",
]


def _ordered_keys(keys: List[str], priority: List[str]) -> List[str]:
    """Priority keys (that are present) first, then the remainder alphabetically."""
    present = set(keys)
    head = [k for k in priority if k in present]
    rest = sorted(k for k in present if k not in priority)
    return head + rest


def scheduler_config_help() -> str:
    """Full help text for the ``--scheduler-config`` keys."""
    res = _scheduler_fields_from_package()
    if res is not None:
        scalar, pkg_descs, defaults = res
        _, schema_descs = _scheduler_schema_fields()
        ordered = _ordered_keys(scalar, _SCHEDULER_PRIORITY)
        rows = [
            (k, defaults.get(k, ""), pkg_descs.get(k) or schema_descs.get(k, ""))
            for k in ordered
        ]
        source = "discovered from the installed openvino_genai package"
    else:
        names, descs = _scheduler_schema_fields()
        rows = [(k, "", descs.get(k, "")) for k in _ordered_keys(names, _SCHEDULER_PRIORITY)]
        source = "openvino_genai unavailable; showing OpenArc SchedulerConfigSchema fields"
    header = (
        "Available --scheduler-config keys (openvino.genai SchedulerConfig)\n"
        f"  source: {source}\n"
        "  Set as a JSON object, e.g. --scheduler-config '{\"cache_size\": 12, \"max_num_seqs\": 1}'.\n"
        "  These bound GPU memory used by the KV cache and batching.\n"
    )
    return header + _render(rows) + "\n"


def runtime_config_help() -> str:
    """Full help text for the ``--runtime-config`` keys."""
    keys = _runtime_keys_from_package()
    if keys is None:
        keys = list(_RUNTIME_KEY_DOCS.keys())
        source = "openvino / GPU device unavailable; showing curated memory-relevant keys"
    else:
        source = "discovered from the installed openvino GPU plugin"
    keys = _ordered_keys(keys, _RUNTIME_PRIORITY)
    rows = [(k, "", _runtime_description(k)) for k in keys]
    header = (
        "Available --runtime-config keys (OpenVINO GPU plugin compile config)\n"
        f"  source: {source}\n"
        "  Set as a JSON object, e.g. --runtime-config '{\"OFFLOAD_RATIO\": 0.05}'.\n"
        "  Memory-relevant keys are documented first; the rest are generic plugin options.\n"
    )
    return header + _render(rows) + "\n"
