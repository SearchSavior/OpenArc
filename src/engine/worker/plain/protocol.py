"""
Wire protocol for the plain-OpenVINO inference worker (stage 3): Kokoro TTS,
Qwen3-ASR, and Qwen3-TTS run in supervised worker subprocesses for
SEGFAULT ISOLATION -- a native crash in OpenVINO inference takes down only
the worker process, which the supervisor respawns with a fresh pipeline,
instead of killing the whole server.

This protocol reuses the shared framing and error types of the GenAI worker
protocol (src.engine.worker.protocol) -- same one-JSON-object-per-line
channel, same line limit, same RemoteWorker* error classes -- and adds the
two request shapes the plain-openvino engines need, which the token-streaming
GenAI protocol never had:

  * RUN        -> RESULT      single result: an audio array (base64 float32
                              samples + sample rate) or a transcription
                              (text + metrics + segments)
  * RUN_STREAM -> ITEM* DONE  audio chunk streaming (each ITEM carries a
                              base64 float32 audio chunk; e.g. Kokoro
                              per-text-chunk audio, Qwen3-TTS PCM chunks)

Audio travels as base64-encoded 1-D float32 sample arrays. The worker process
does the inference (the segfault-prone part); the server process keeps the
WAV encoding (infer_kokoro / infer_qwen3_tts), so the public response shapes
are unchanged.
"""

from __future__ import annotations

# Shared framing, limits, and error types: one protocol family, so the names
# and bytes come from a single place.
from src.engine.worker.protocol import (
    PROTOCOL_LINE_LIMIT,
    PROTOCOL_VERSION,
    OP_CANCEL,
    OP_LOAD,
    OP_PING,
    OP_UNLOAD,
    MSG_BYE,
    MSG_CANCEL_ACK,
    MSG_DONE,
    MSG_ERROR,
    MSG_FATAL,
    MSG_ITEM,
    MSG_LOAD_ERROR,
    MSG_LOAD_OK,
    MSG_READY,
    MSG_RESULT,
    ProtocolError,
    RemoteWorkerDeadError,
    RemoteWorkerError,
    RemoteWorkerLoadError,
    decode_request,
    decode_response,
    encode,
    encode_response,
    is_non_recoverable_error,
    serialize_error,
)

__all__ = [
    "PROTOCOL_LINE_LIMIT",
    "PROTOCOL_VERSION",
    "OP_CANCEL",
    "OP_LOAD",
    "OP_PING",
    "OP_UNLOAD",
    "MSG_BYE",
    "MSG_CANCEL_ACK",
    "MSG_DONE",
    "MSG_ERROR",
    "MSG_FATAL",
    "MSG_ITEM",
    "MSG_LOAD_ERROR",
    "MSG_LOAD_OK",
    "MSG_READY",
    "MSG_RESULT",
    "ProtocolError",
    "RemoteWorkerDeadError",
    "RemoteWorkerError",
    "RemoteWorkerLoadError",
    "decode_request",
    "decode_response",
    "encode",
    "encode_response",
    "is_non_recoverable_error",
    "serialize_error",
    "OP_RUN",
    "OP_RUN_STREAM",
]

# --- plain-OpenVINO-specific ops -------------------------------------------------
# payload: req_id, request_id, gen_config (the engine's gen-config JSON).
# RUN: the worker answers with one MSG_RESULT (a JSON dict).
OP_RUN = "RUN"
# RUN_STREAM: the worker answers with MSG_ITEM per audio chunk, then MSG_DONE.
OP_RUN_STREAM = "RUN_STREAM"
