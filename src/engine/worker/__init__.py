"""Out-of-process inference worker machinery (stage 1: OpenVINO GenAI VLM).

The main server process never builds an OpenVINO pipeline for these models;
each one runs in a supervised child process (see docs/worker-processes.md).
"""

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
]
