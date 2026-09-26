"""
Entry point of an OpenArc inference worker process.

One of these runs per loaded model. It owns the OpenVINO / openvino.genai
pipeline for that model -- the main server process never builds one.
Communication with the supervisor is one JSON object per line over stdin
(commands) / stdout (responses); all logging and OpenVINO/OpenCL diagnostics
go to stderr, which the supervisor forwards to the openarc log.

Exit codes: 0 = clean unload / parent gone; 1 = fatal (MSG_FATAL was sent)
or unhandled crash. The supervisor treats every non-clean exit as "the
process is gone" and decides whether to respawn a fresh one.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from src.engine.worker import protocol as proto
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.modeling.contract_whisper import OVGenAI_WhisperGenConfig
from src.server.schemas.registration import ModelLoadConfig, ModelType

logger = logging.getLogger("openarc.worker")

# Engines the worker process knows how to run (stage 1+2: OpenVINO GenAI
# VLM/LLM/Whisper). Everything else stays in the server process for now.
_WORKER_MODEL_TYPES = (ModelType.VLM, ModelType.LLM, ModelType.WHISPER)

_GEN_CONFIG_CLASSES = {
    ModelType.LLM: OVGenAI_GenConfig,
    ModelType.VLM: OVGenAI_GenConfig,
    ModelType.WHISPER: OVGenAI_WhisperGenConfig,
}


def _make_model(load_config: ModelLoadConfig) -> Any:
    """Build the model object inside the worker process (real or stub)."""
    if load_config.model_type not in _WORKER_MODEL_TYPES:
        raise ValueError(
            f"model type {load_config.model_type.value!r} is not supported "
            f"in the worker process (supported: {', '.join(t.value for t in _WORKER_MODEL_TYPES)})"
        )
    if os.environ.get("OPENARC_WORKER_STUB", "").strip().lower() in _STUB_ON:
        return _StubModel(load_config)
    # Imported here (and only here) so the stub path -- and unit tests in
    # general -- never pull OpenVINO into the process.
    if load_config.model_type == ModelType.VLM:
        from src.engine.ov_genai.vlm import OVGenAI_VLM

        return OVGenAI_VLM(load_config)
    if load_config.model_type == ModelType.LLM:
        from src.engine.ov_genai.llm import OVGenAI_LLM

        return OVGenAI_LLM(load_config)
    from src.engine.ov_genai.whisper import OVGenAI_Whisper

    return OVGenAI_Whisper(load_config)


def _configure_logging() -> None:
    """Point the root logger at stderr; stdout is the protocol channel."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [worker pid=%(process)d] %(name)s: %(message)s"
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


_STUB_ON = ("1", "true", "yes", "on")


def _worker_line_limit() -> int:
    """The IPC line limit (bytes) this worker's stdin reader accepts.

    The supervisor exports it in the environment at spawn time, because the
    stdin reader is created before the LOAD command (which carries the model
    config) arrives. It reflects the model's ``worker_line_limit`` setting,
    or the protocol default when the model does not override it.
    """
    raw = os.environ.get("OPENARC_WORKER_LINE_LIMIT", "").strip()
    if not raw:
        return proto.PROTOCOL_LINE_LIMIT
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"invalid OPENARC_WORKER_LINE_LIMIT {raw!r}; using the protocol default"
        )
        return proto.PROTOCOL_LINE_LIMIT


class _StubModel:
    """
    Test double for the OVGenAI models (VLM/LLM/Whisper), selected by
    OPENARC_WORKER_STUB=1 so the whole spawn/IPC/respawn machinery can be
    exercised in unit tests without OpenVINO, a GPU, or model files. It
    mirrors the yield contracts: non-streaming generate_type yields
    (metrics dict, text); streaming yields text chunks then a metrics dict;
    transcribe yields (metrics dict, text).

    Prompt keywords (or, for transcribe, the decoded audio payload) drive
    behaviour:
      CRASH -> os._exit(139)  (simulates a native crash / GPU wedge)
      FATAL -> raises a CL_OUT_OF_RESOURCES-looking error (non-recoverable:
               the worker reports MSG_FATAL and exits)
      RAISE -> yields one token then raises (recoverable: MSG_ERROR, the
               worker stays up)
      SLOW  -> per-token delay (OPENARC_WORKER_STUB_TOKEN_DELAY, default 0.05s)
    """

    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self._cancel_events: Dict[str, asyncio.Event] = {}

    def load_model(self, loader: ModelLoadConfig) -> None:
        if os.environ.get("OPENARC_WORKER_STUB_FAIL_LOAD", "").strip().lower() in _STUB_ON:
            raise RuntimeError("stub load failure")
        delay = float(os.environ.get("OPENARC_WORKER_STUB_LOAD_DELAY", "0.05"))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _payload_text(gen_config: Any) -> str:
        """The control text: the prompt/messages for generate, the decoded
        (text-encoded) audio payload for transcribe."""
        if isinstance(gen_config, OVGenAI_WhisperGenConfig):
            try:
                return base64.b64decode(gen_config.audio_base64).decode("utf-8", "replace")
            except Exception:
                return ""
        if gen_config.prompt:
            return gen_config.prompt
        parts: List[str] = []
        for message in gen_config.messages or []:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
        return " ".join(parts)

    async def generate_type(self, gen_config: OVGenAI_GenConfig) -> Any:
        text = self._payload_text(gen_config)
        if "CRASH" in text:
            os._exit(139)
        tokens = [t for t in text.split() if t not in ("CRASH", "FATAL", "RAISE", "SLOW")] or ["ok"]
        if "FATAL" in text:
            yield "partial"
            raise RuntimeError("CL_OUT_OF_RESOURCES (stub)")
        if "RAISE" in text:
            yield "first"
            raise RuntimeError("stub recoverable error")
        if gen_config.request_id is not None:
            self._cancel_events[gen_config.request_id] = asyncio.Event()
        if gen_config.stream:
            ev = self._cancel_events.get(gen_config.request_id) if gen_config.request_id else None
            delay = float(os.environ.get("OPENARC_WORKER_STUB_TOKEN_DELAY", "0.05"))
            for token in tokens:
                if ev is not None and ev.is_set():
                    return  # cancelled: end the stream without metrics
                yield token
                if delay > 0:
                    await asyncio.sleep(delay)
            if ev is not None and ev.is_set():
                return
            yield {"new_token": len(tokens), "stream": True}
        else:
            yield {"new_token": len(tokens), "stream": False}
            yield " ".join(tokens)

    async def transcribe(self, gen_config: OVGenAI_WhisperGenConfig) -> Any:
        text = self._payload_text(gen_config)
        if "CRASH" in text:
            os._exit(139)
        if "FATAL" in text:
            raise RuntimeError("CL_OUT_OF_RESOURCES (stub)")
        if "RAISE" in text:
            raise RuntimeError("stub recoverable error")
        # Same yield contract as OVGenAI_Whisper.transcribe: metrics dict,
        # then the transcribed text.
        yield {"num_generated_tokens": 3, "throughput_tokens_per_sec": 1.0}
        yield "stub transcript"

    async def cancel(self, request_id: str) -> bool:
        ev = self._cancel_events.get(request_id)
        if ev is not None:
            ev.set()
            return True
        return False


class _Worker:
    """The child-side protocol loop.

    The stdin read loop never blocks on generation: a GENERATE is dispatched
    to a background task, so PING / CANCEL / UNLOAD are always processed even
    while a pipeline is running.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name = ""
        self.model_type: Optional[ModelType] = None
        self._active_gen: Optional[asyncio.Task] = None
        self._active_request_id: Optional[str] = None
        self._send_lock = asyncio.Lock()

    async def send(self, data: bytes) -> None:
        async with self._send_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        # Protocol lines can be very large (a VLM chat request with base64
        # images is one multi-MB JSON line); the default 64 KiB readline
        # limit would raise "Separator is found, but chunk is longer than
        # limit" and kill the worker on the first real request. The limit
        # itself is per-model (see ModelLoadConfig.worker_line_limit,
        # exported by the supervisor as OPENARC_WORKER_LINE_LIMIT).
        reader = asyncio.StreamReader(limit=_worker_line_limit())
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            try:
                line = await reader.readline()
            except ValueError as e:
                # A line even beyond PROTOCOL_LINE_LIMIT: report it clearly
                # and take the process down (the supervisor respawns).
                logger.error(f"protocol line from parent exceeds the limit: {e}")
                try:
                    await self.send(
                        proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e))
                    )
                except Exception:
                    pass
                os._exit(1)
            if not line:
                logger.info("parent closed stdin; exiting")
                break
            try:
                msg = proto.decode_request(line)
            except proto.ProtocolError as e:
                logger.warning(f"bad protocol message from parent: {e}")
                continue
            try:
                await self._handle(msg)
            except Exception:
                logger.exception("failed to handle protocol message")
        if self._active_gen is not None and not self._active_gen.done():
            self._active_gen.cancel()

    # Overridable hooks so protocol variants (e.g. the plain-OpenVINO worker)
    # can subclass this loop without duplicating it.
    def _build_model(self, load_config: ModelLoadConfig) -> Any:
        return _make_model(load_config)

    def _gen_config_cls(self) -> Optional[Any]:
        if self.model_type is None:
            return None
        return _GEN_CONFIG_CLASSES.get(self.model_type)

    async def _handle(self, msg: Dict[str, Any]) -> None:
        op = msg["op"]
        if op == proto.OP_PING:
            await self.send(proto.encode_response(proto.MSG_READY, req_id=msg["req_id"]))
        elif op == proto.OP_LOAD:
            asyncio.create_task(self._do_load(msg), name="ovworker-load")
        elif op == proto.OP_GENERATE:
            await self._start_run(msg, "generate_type")
        elif op == proto.OP_TRANSCRIBE:
            await self._start_run(msg, "transcribe")
        elif op == proto.OP_CANCEL:
            ok = False
            if self.model is not None and hasattr(self.model, "cancel"):
                try:
                    ok = bool(await self.model.cancel(msg["request_id"]))
                except Exception:
                    logger.exception("worker-side cancel failed")
            await self.send(proto.encode_response(proto.MSG_CANCEL_ACK, req_id=msg["req_id"], ok=ok))
        elif op == proto.OP_UNLOAD:
            await self.send(proto.encode_response(proto.MSG_BYE, req_id=msg["req_id"]))
            await self._shutdown()
        else:
            await self._handle_custom_op(msg)

    async def _handle_custom_op(self, msg: Dict[str, Any]) -> None:
        """Extension point for ops this protocol does not know (subclasses)."""
        logger.warning(f"unknown protocol op: {msg.get('op')!r}")

    async def _do_load(self, msg: Dict[str, Any]) -> None:
        try:
            load_config = ModelLoadConfig.model_validate_json(msg["config"])
            self.model_name = load_config.model_name
            self.model_type = load_config.model_type
            model = self._build_model(load_config)
            logger.info(f"[{load_config.model_name}] building pipeline on {load_config.device} ...")
            await asyncio.to_thread(model.load_model, load_config)
            self.model = model
            logger.info(f"[{load_config.model_name}] pipeline ready")
            await self.send(proto.encode_response(proto.MSG_LOAD_OK, req_id=msg["req_id"]))
        except Exception as e:
            logger.error(f"[{self.model_name or 'load'}] pipeline load failed", exc_info=True)
            if proto.is_non_recoverable_error(e):
                # The device is wedged: this process's ov::Core is poisoned
                # and no in-process retry can fix it -- take the process down
                # and let the supervisor spawn a fresh one.
                await self.send(proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e)))
                os._exit(1)
            await self.send(
                proto.encode_response(
                    proto.MSG_LOAD_ERROR, req_id=msg["req_id"], error=proto.serialize_error(e)
                )
            )

    def _validate_gen_config(self, raw: Dict[str, Any]) -> Any:
        """Validate the gen_config payload against the loaded engine's contract."""
        if self.model_type is None:
            raise ValueError("no model loaded")
        gen_config_cls = self._gen_config_cls()
        if gen_config_cls is None:
            raise ValueError(f"no gen config contract for model type {self.model_type!r}")
        raw = dict(raw)
        # Drop explicit nulls on fields whose declared default is None: the
        # parent serialises whatever the in-memory object carries, and
        # pydantic does not validate defaults, so a field declared
        # non-optional with a None default arrives as null and would fail
        # validation against its declared type even though the in-memory
        # object holds exactly that value. Removing the key lets the default
        # apply. Required fields (no default) are left untouched, so a null
        # there still fails loudly.
        for name, field_info in gen_config_cls.model_fields.items():
            if field_info.default is None and raw.get(name) is None:
                raw.pop(name)
        return gen_config_cls.model_validate(raw)

    async def _check_run_preconditions(self, msg: Dict[str, Any]) -> Optional[bytes]:
        """Shared guards for starting a run (reused by protocol variants):
        an error response line to send, or None when the run may start."""
        req_id = msg["req_id"]
        if self.model is None:
            return proto.encode_response(
                proto.MSG_ERROR,
                req_id=req_id,
                error={"type": "NotLoaded", "message": "no model loaded"},
            )
        if self._active_gen is not None and not self._active_gen.done():
            return proto.encode_response(
                proto.MSG_ERROR,
                req_id=req_id,
                error={"type": "Busy", "message": "an inference is already in progress"},
            )
        return None

    async def _start_run(self, msg: Dict[str, Any], method_name: str) -> None:
        err = await self._check_run_preconditions(msg)
        if err is not None:
            await self.send(err)
            return
        self._active_gen = asyncio.create_task(
            self._run_inference(msg, method_name), name=f"ovworker-gen-{msg['req_id'][:8]}"
        )

    async def _report_run_error(self, e: BaseException, req_id: str, method_name: str) -> None:
        """The shared error ladder for failed runs (reused by protocol
        variants): recoverable -> MSG_ERROR (worker stays up); non-recoverable
        (wedged device / native failure) -> MSG_FATAL + exit so the supervisor
        respawns a clean process."""
        logger.error(f"[{self.model_name}] {method_name} failed", exc_info=True)
        if proto.is_non_recoverable_error(e):
            try:
                await self.send(
                    proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e))
                )
            except Exception:
                pass
            os._exit(1)
        await self.send(
            proto.encode_response(proto.MSG_ERROR, req_id=req_id, error=proto.serialize_error(e))
        )

    async def _run_inference(self, msg: Dict[str, Any], method_name: str) -> None:
        req_id = msg["req_id"]
        self._active_request_id = msg.get("request_id")
        try:
            gen_config = self._validate_gen_config(json.loads(msg["gen_config"]))
            run = getattr(self.model, method_name)(gen_config)
            async for item in run:
                await self.send(proto.encode_response(proto.MSG_ITEM, req_id=req_id, item=item))
            await self.send(proto.encode_response(proto.MSG_DONE, req_id=req_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._report_run_error(e, req_id, method_name)
        finally:
            self._active_request_id = None
            self._active_gen = None

    async def _shutdown(self) -> None:
        """Finish an in-flight generation (giving it a grace period), exit 0."""
        if self._active_gen is not None and not self._active_gen.done():
            if (
                self._active_request_id
                and self.model is not None
                and hasattr(self.model, "cancel")
            ):
                try:
                    await self.model.cancel(self._active_request_id)
                except Exception:
                    pass
            _, pending = await asyncio.wait({self._active_gen}, timeout=5.0)
            for task in pending:
                task.cancel()
        sys.exit(0)


def main() -> int:
    """Run the worker protocol loop. Returns the process exit code."""
    _configure_logging()
    worker = _Worker()
    try:
        asyncio.run(worker.run())
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        logger.error(f"worker main loop crashed: {e}", exc_info=True)
        try:
            sys.stdout.buffer.write(
                proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e))
            )
            sys.stdout.buffer.flush()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
