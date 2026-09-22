"""Unit tests for the worker wire protocol (src.engine.worker.protocol)."""

import json

import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto


def test_encode_decode_request_roundtrip() -> None:
    line = proto.encode(proto.OP_GENERATE, req_id="r1", request_id="req-9", gen_config='{"prompt": "hi"}')
    msg = proto.decode_request(line)
    assert msg["v"] == proto.PROTOCOL_VERSION
    assert msg["op"] == proto.OP_GENERATE
    assert msg["req_id"] == "r1"
    assert msg["request_id"] == "req-9"
    assert msg["gen_config"] == '{"prompt": "hi"}'


def test_encode_decode_response_roundtrip() -> None:
    item = {"new_token": 42, "stream": True}
    line = proto.encode_response(proto.MSG_ITEM, req_id="r1", item=item)
    msg = proto.decode_response(line)
    assert msg["type"] == proto.MSG_ITEM
    assert msg["item"] == item


def test_every_op_and_msg_roundtrips() -> None:
    ops = [proto.OP_LOAD, proto.OP_GENERATE, proto.OP_TRANSCRIBE, proto.OP_CANCEL, proto.OP_PING, proto.OP_UNLOAD]
    msgs = [
        proto.MSG_READY,
        proto.MSG_LOAD_OK,
        proto.MSG_LOAD_ERROR,
        proto.MSG_ITEM,
        proto.MSG_DONE,
        proto.MSG_ERROR,
        proto.MSG_FATAL,
        proto.MSG_BYE,
        proto.MSG_CANCEL_ACK,
    ]
    for op in ops:
        assert proto.decode_request(proto.encode(op, req_id="x"))["op"] == op
    for mtype in msgs:
        assert proto.decode_response(proto.encode_response(mtype, req_id="x"))["type"] == mtype


def test_decode_request_rejects_response_shape() -> None:
    line = (json.dumps({"v": proto.PROTOCOL_VERSION, "type": "DONE"}) + "\n").encode()
    with pytest.raises(proto.ProtocolError):
        proto.decode_request(line)


def test_decode_response_rejects_request_shape() -> None:
    line = (json.dumps({"v": proto.PROTOCOL_VERSION, "op": "PING"}) + "\n").encode()
    with pytest.raises(proto.ProtocolError):
        proto.decode_response(line)


def test_decode_rejects_bad_version() -> None:
    line = (json.dumps({"v": 99, "op": "LOAD"}) + "\n").encode()
    with pytest.raises(proto.ProtocolError, match="version"):
        proto.decode_request(line)


def test_decode_rejects_garbage() -> None:
    with pytest.raises(proto.ProtocolError):
        proto.decode_request(b"this is not json\n")


def test_decode_accepts_bytes_and_str() -> None:
    line = proto.encode(proto.OP_PING, req_id="p")
    assert proto.decode_request(line)["op"] == proto.OP_PING
    assert proto.decode_request(line.decode("utf-8"))["op"] == proto.OP_PING


def test_serialize_error() -> None:
    err = proto.serialize_error(ValueError("boom"))
    assert err == {"type": "ValueError", "message": "boom"}
    # Must be JSON-safe.
    json.dumps(err)


def test_is_non_recoverable_markers() -> None:
    assert proto.is_non_recoverable_error(RuntimeError("CL_OUT_OF_RESOURCES"))
    assert proto.is_non_recoverable_error(RuntimeError("CL_INVALID_EVENT"))
    assert proto.is_non_recoverable_error(RuntimeError("CL_OUT_OF_HOST_MEMORY"))
    assert proto.is_non_recoverable_error(RuntimeError("CL_DEVICE_LOST"))
    assert proto.is_non_recoverable_error(RuntimeError("CL_DRIVER_ERROR"))
    assert proto.is_non_recoverable_error(RuntimeError("could not execute a primitive"))
    assert proto.is_non_recoverable_error(RuntimeError("[GPU] ProgramBuilder build failed!"))
    # Case-insensitive.
    assert proto.is_non_recoverable_error(RuntimeError("Cl_Out_Of_Resources"))


def test_is_recoverable_errors_are_not_flagged() -> None:
    assert not proto.is_non_recoverable_error(RuntimeError("CL_INVALID_VALUE: bad property"))
    assert not proto.is_non_recoverable_error(ValueError("native vision tag count mismatch"))
    assert not proto.is_non_recoverable_error(RuntimeError("stub recoverable error"))
    assert not proto.is_non_recoverable_error(KeyError("missing model file"))


def test_remote_worker_error_carries_original_type() -> None:
    exc = proto.RemoteWorkerError("CL_OUT_OF_RESOURCES", original_type="RuntimeError")
    assert str(exc) == "CL_OUT_OF_RESOURCES"
    assert exc.original_type == "RuntimeError"
    assert isinstance(exc, Exception)
    assert not isinstance(exc, proto.RemoteWorkerDeadError)
