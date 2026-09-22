"""OpenArc inference engines.

The heavy engine classes (OpenVINO GenAI, Kokoro, ...) are exported
lazily (PEP 562): ``from src.engine import OVGenAI_LLM`` keeps working, but
merely importing ``src.engine`` -- e.g. via its ``src.engine.worker``
subpackage, which the inference worker subprocesses import at startup -- no
longer drags torch / openvino_genai / transformers / kokoro into the process.
That keeps worker-process startup (and every respawn) cheap.
"""

import importlib

_LAZY_EXPORTS = {
    "OVGenAI_LLM": "src.engine.ov_genai.llm",
    "OVGenAI_VLM": "src.engine.ov_genai.vlm",
    "OVGenAI_Whisper": "src.engine.ov_genai.whisper",
    "OV_Kokoro": "src.engine.openvino.kokoro",
    "ChunkStreamer": "src.engine.ov_genai.streamers",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
