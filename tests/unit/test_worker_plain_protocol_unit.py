"""Unit tests for the plain-OpenVINO worker protocol (stage 3): framing
reuse from the GenAI protocol, the RUN / RUN_STREAM / RESULT shapes, and the
audio (de)serialization that crosses the pipe."""

import numpy as np

from src.engine.worker import protocol as shared
from src.engine.worker.plain import protocol as p_proto


def test_shared_names_are_byte_identical() -> None:
    """The plain protocol reuses the shared framing: the reused ops/messages
    must be the exact same strings, so both sides agree byte-for-byte."""
    assert p_proto.OP_LOAD == shared.OP_LOAD
    assert p_proto.OP_CANCEL == shared.OP_CANCEL
    assert p_proto.OP_PING == shared.OP_PING
    assert p_proto.OP_UNLOAD == shared.OP_UNLOAD
    assert p_proto.MSG_ITEM == shared.MSG_ITEM
    assert p_proto.MSG_DONE == shared.MSG_DONE
    assert p_proto.MSG_RESULT == shared.MSG_RESULT
    assert p_proto.PROTOCOL_VERSION == shared.PROTOCOL_VERSION
    assert p_proto.PROTOCOL_LINE_LIMIT == shared.PROTOCOL_LINE_LIMIT


def test_run_ops_are_distinct() -> None:
    all_shared = {
        shared.OP_LOAD,
        shared.OP_GENERATE,
        shared.OP_TRANSCRIBE,
        shared.OP_CANCEL,
        shared.OP_PING,
        shared.OP_UNLOAD,
    }
    assert p_proto.OP_RUN not in all_shared
    assert p_proto.OP_RUN_STREAM not in all_shared
    assert p_proto.OP_RUN != p_proto.OP_RUN_STREAM


def test_run_request_roundtrip() -> None:
    line = p_proto.encode(
        p_proto.OP_RUN, req_id="r1", request_id="req-9", gen_config='{"input": "hi"}'
    )
    msg = p_proto.decode_request(line)
    assert msg["v"] == p_proto.PROTOCOL_VERSION
    assert msg["op"] == p_proto.OP_RUN
    assert msg["req_id"] == "r1"
    assert msg["request_id"] == "req-9"
    assert msg["gen_config"] == '{"input": "hi"}'


def test_run_stream_request_roundtrip() -> None:
    line = p_proto.encode(
        p_proto.OP_RUN_STREAM, req_id="r2", gen_config='{"input": "stream me"}'
    )
    msg = p_proto.decode_request(line)
    assert msg["op"] == p_proto.OP_RUN_STREAM
    assert msg["req_id"] == "r2"


def test_result_response_roundtrip() -> None:
    payload = {"text": "hello", "metrics": {"a": 1}, "segments": [{"id": 0}]}
    line = p_proto.encode_response(p_proto.MSG_RESULT, req_id="r1", result=payload)
    msg = p_proto.decode_response(line)
    assert msg["type"] == p_proto.MSG_RESULT
    assert msg["req_id"] == "r1"
    assert msg["result"] == payload


def test_audio_result_response_roundtrip() -> None:
    payload = {"audio_b64": "AQID", "samples": 1, "sample_rate": 24000}
    line = p_proto.encode_response(p_proto.MSG_RESULT, req_id="r3", result=payload)
    msg = p_proto.decode_response(line)
    assert msg["result"]["sample_rate"] == 24000
    assert msg["result"]["samples"] == 1


def test_audio_roundtrip_numpy() -> None:
    from src.engine.worker.plain.worker_client import _decode_audio_b64
    from src.engine.worker.plain.worker_process import _to_float32_b64

    arr = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    b64, samples = _to_float32_b64(arr)
    assert samples == 4
    decoded = _decode_audio_b64({"audio_b64": b64})
    assert decoded.dtype == np.float32
    assert decoded.shape == (4,)
    assert np.allclose(decoded, arr)


def test_audio_roundtrip_torch() -> None:
    """Kokoro chunks carry torch tensors on the worker side."""
    import torch

    from src.engine.worker.plain.worker_client import _decode_audio_b64
    from src.engine.worker.plain.worker_process import _to_float32_b64

    tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    b64, samples = _to_float32_b64(tensor)
    assert samples == 3
    decoded = _decode_audio_b64({"audio_b64": b64})
    assert np.allclose(decoded, [0.1, 0.2, 0.3])
