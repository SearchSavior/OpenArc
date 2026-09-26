"""
Plain-OpenVINO inference worker package (stage 3).

Kokoro TTS, Qwen3-ASR, and Qwen3-TTS run in supervised worker subprocesses
for segfault isolation. The package contains:

  * protocol       -- the plain-OpenVINO wire protocol (RUN / RUN_STREAM)
  * supervisor     -- PlainWorkerSupervisor (GenAI supervisor + entry point)
  * worker_process -- the CHILD entry point (never imported by the server)
  * worker_client  -- the server-side facades (RemoteOV_Kokoro, ...)
"""

from src.engine.worker.plain.protocol import (
    OP_RUN,
    OP_RUN_STREAM,
    MSG_RESULT,
    RemoteWorkerDeadError,
    RemoteWorkerError,
    RemoteWorkerLoadError,
)
from src.engine.worker.plain.supervisor import PlainWorkerSupervisor
from src.engine.worker.plain.worker_client import (
    RemoteOV_Kokoro,
    RemoteOVQwen3ASR,
    RemoteOVQwen3TTS,
)

__all__ = [
    "OP_RUN",
    "OP_RUN_STREAM",
    "MSG_RESULT",
    "RemoteWorkerDeadError",
    "RemoteWorkerError",
    "RemoteWorkerLoadError",
    "PlainWorkerSupervisor",
    "RemoteOV_Kokoro",
    "RemoteOVQwen3ASR",
    "RemoteOVQwen3TTS",
]
