"""Out-of-process inference worker machinery.

Stage 1+2: OpenVINO GenAI (VLM/LLM/Whisper) -- token-streaming protocol.
Stage 3: plain OpenVINO (Kokoro/Qwen3-ASR/Qwen3-TTS) -- single-result and
audio-streaming protocol (src.engine.worker.plain).

The main server process never builds an OpenVINO pipeline for these models;
each one runs in a supervised child process (see docs/worker-processes.md).
"""

from src.engine.worker.plain import (
    PlainWorkerSupervisor,
    RemoteOV_Kokoro,
    RemoteOVQwen3ASR,
    RemoteOVQwen3TTS,
)
from src.engine.worker.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    RemoteWorkerDeadError,
    RemoteWorkerError,
    RemoteWorkerLoadError,
    is_non_recoverable_error,
    serialize_error,
)
from src.engine.worker.supervisor import EOF, WorkerSupervisor
from src.engine.worker.worker_client import (
    RemoteOVGenAI,
    RemoteOVGenAI_LLM,
    RemoteOVGenAI_VLM,
    RemoteOVGenAI_Whisper,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ProtocolError",
    "RemoteWorkerDeadError",
    "RemoteWorkerError",
    "RemoteWorkerLoadError",
    "is_non_recoverable_error",
    "serialize_error",
    "EOF",
    "WorkerSupervisor",
    "RemoteOVGenAI",
    "RemoteOVGenAI_VLM",
    "RemoteOVGenAI_LLM",
    "RemoteOVGenAI_Whisper",
    "PlainWorkerSupervisor",
    "RemoteOV_Kokoro",
    "RemoteOVQwen3ASR",
    "RemoteOVQwen3TTS",
]
