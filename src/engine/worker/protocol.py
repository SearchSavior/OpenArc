"""
Wire protocol between the OpenArc main process and its inference worker
subprocesses.

Each model that runs in a worker process (currently OpenVINO GenAI VLMs) is
served by exactly one dedicated child process. The two sides talk over the
child's stdin/stdout using one JSON object per line:

    parent -> worker:  {"v": 1, "op": "<OP>", ...}
    worker -> parent:  {"v": 1, "type": "<MSG>", ...}

stdout is reserved for protocol traffic; everything the worker would like to
say that is not protocol (logging, OpenVINO/OpenCL diagnostics, Python
tracebacks) must go to stderr, where the parent forwards it to openarc.log.

Why a process boundary at all: openvino_genai pipelines share a
process-wide singleton ``ov::Core``. When the GPU plugin wedges
(CL_OUT_OF_RESOURCES, device loss, ...), no in-process unload or recompile
can fix it -- the poisoned Core outlives the model. Killing the worker
process and starting a fresh one is the only way to get a clean Core, which
is what makes unload/load reliable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

PROTOCOL_VERSION = 1

# Maximum size of one protocol line, in bytes.
#
# The protocol is one JSON object per line, and a request line can be large:
# a VLM chat request carries the full conversation (including base64 images)
# inside gen_config, so lines of tens of MB are normal. asyncio's
# StreamReader.readline() rejects lines longer than its limit (64 KiB by
# default) with "Separator is found, but chunk is longer than limit", which
# would kill the worker on every real chat request. Every side must read the
# protocol channel with a reader created/configured for this limit.
PROTOCOL_LINE_LIMIT = 256 * 1024 * 1024

# --- parent -> worker ops ---------------------------------------------------
OP_LOAD = "LOAD"        # payload: req_id, config (ModelLoadConfig JSON)
OP_GENERATE = "GENERATE"  # payload: req_id, request_id, gen_config (OVGenAI_GenConfig JSON)
OP_TRANSCRIBE = "TRANSCRIBE"  # payload: req_id, gen_config (OVGenAI_WhisperGenConfig JSON)
OP_CANCEL = "CANCEL"    # payload: req_id, request_id
OP_PING = "PING"        # payload: req_id
OP_UNLOAD = "UNLOAD"    # payload: req_id

# --- worker -> parent message types ------------------------------------------
MSG_READY = "READY"            # ack of PING: worker is alive and responsive
MSG_LOAD_OK = "LOAD_OK"        # pipeline built and ready
MSG_LOAD_ERROR = "LOAD_ERROR"  # payload: req_id, error -- clean load failure, worker stays up
MSG_ITEM = "ITEM"              # payload: req_id, item -- one item yielded by generate_type (str chunk or dict)
MSG_DONE = "DONE"              # payload: req_id -- generation finished without error
MSG_RESULT = "RESULT"          # payload: req_id, result -- single-result run finished (plain-OpenVINO protocol)
MSG_ERROR = "ERROR"            # payload: req_id, error -- per-request error, worker stays up
MSG_FATAL = "FATAL"            # payload: error -- worker is exiting; process-wide failure
MSG_BYE = "BYE"                # payload: req_id -- ack of UNLOAD; worker is exiting cleanly
MSG_CANCEL_ACK = "CANCEL_ACK"  # payload: req_id, ok


class ProtocolError(ValueError):
    """A line on the protocol channel is not a valid protocol message."""


def _parse(line: Union[bytes, str], key: str) -> Dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON on protocol channel: {e}") from e
    if not isinstance(msg, dict):
        raise ProtocolError(f"protocol message is not an object: {line[:200]!r}")
    if msg.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {msg.get('v')!r} (want {PROTOCOL_VERSION})"
        )
    if key not in msg:
        raise ProtocolError(f"protocol message is missing '{key}': {line[:200]!r}")
    return msg


def encode(op: str, **payload: Any) -> bytes:
    """Encode a parent -> worker request as one protocol line."""
    return (json.dumps({"v": PROTOCOL_VERSION, "op": op, **payload}) + "\n").encode("utf-8")


def encode_response(type_: str, **payload: Any) -> bytes:
    """Encode a worker -> parent message as one protocol line."""
    return (json.dumps({"v": PROTOCOL_VERSION, "type": type_, **payload}) + "\n").encode("utf-8")


def decode_request(line: Union[bytes, str]) -> Dict[str, Any]:
    """Parse a parent -> worker line. Raises ProtocolError on malformed input."""
    return _parse(line, "op")


def decode_response(line: Union[bytes, str]) -> Dict[str, Any]:
    """Parse a worker -> parent line. Raises ProtocolError on malformed input."""
    return _parse(line, "type")


def serialize_error(exc: BaseException) -> Dict[str, Any]:
    """Reduce an exception to a JSON-safe {type, message} payload."""
    return {"type": type(exc).__name__, "message": str(exc)}


class RemoteWorkerError(Exception):
    """A failure that happened inside (or to) the worker process.

    ``original_type`` carries the exception class name from the worker side
    when one was reported, so log lines can show what actually raised.
    """

    def __init__(self, message: str, *, original_type: Optional[str] = None):
        super().__init__(message)
        self.original_type = original_type


class RemoteWorkerLoadError(RemoteWorkerError):
    """The worker process failed to load the model."""


class RemoteWorkerDeadError(RemoteWorkerError):
    """The worker process is gone (crashed, killed, unloaded) or not (re)started.

    Distinct from a plain RemoteWorkerError on purpose: a dead worker is the
    supervisor's problem (it respawns within budget, or unloads the model once
    the budget is exhausted), so the registry must NOT also trigger an unload.
    """


# OpenCL / driver conditions that poison the process-wide ov::Core: once one
# of these happens, no in-process model unload or recompile can recover the
# device state, and the only clean remedy is a fresh process (fresh Core).
_NON_RECOVERABLE_MARKERS = (
    "cl_out_of_resources",
    "cl_out_of_host_memory",
    "cl_driver_error",
    "cl_device_lost",
    "cl_invalid_event",
    "could not execute a primitive",
    "programbuilder build failed",
)


def is_non_recoverable_error(exc: BaseException) -> bool:
    """True when an exception looks like a wedged-device / driver failure.

    Used by the worker process to decide between a per-request ERROR (the
    worker stays up) and a FATAL that takes the whole process down so the
    supervisor can respawn a clean one.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _NON_RECOVERABLE_MARKERS)
