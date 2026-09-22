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
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from src.engine.worker import protocol as proto
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger("openarc.worker")


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


class _StubVLM:
    """
    Test double for OVGenAI_VLM, selected by OPENARC_WORKER_STUB=1 so the
    whole spawn/IPC/respawn machinery can be exercised in unit tests without
    OpenVINO, a GPU, or model files. It mirrors OVGenAI_VLM's yield contract:
    non-streaming yields (metrics dict, text); streaming yields text chunks
    then a metrics dict.

    Prompt keywords drive behaviour:
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
    def _prompt_text(gen_config: OVGenAI_GenConfig) -> str:
        if gen_config.prompt:
            return gen_config.prompt
        parts: List[str] = []
        for message in gen_config.messages or []:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
        return " ".join(parts)

    async def generate_type(self, gen_config: OVGenAI_GenConfig) -> Any:
        text = self._prompt_text(gen_config)
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
        self._active_gen: Optional[asyncio.Task] = None
        self._active_request_id: Optional[str] = None
        self._send_lock = asyncio.Lock()

    async def send(self, data: bytes) -> None:
        async with self._send_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            line = await reader.readline()
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

    async def _handle(self, msg: Dict[str, Any]) -> None:
        op = msg["op"]
        if op == proto.OP_PING:
            await self.send(proto.encode_response(proto.MSG_READY, req_id=msg["req_id"]))
        elif op == proto.OP_LOAD:
            asyncio.create_task(self._do_load(msg), name="ovworker-load")
        elif op == proto.OP_GENERATE:
            await self._do_generate(msg)
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

    async def _do_load(self, msg: Dict[str, Any]) -> None:
        try:
            load_config = ModelLoadConfig.model_validate_json(msg["config"])
            self.model_name = load_config.model_name
            if os.environ.get("OPENARC_WORKER_STUB", "").strip().lower() in _STUB_ON:
                model = _StubVLM(load_config)
            else:
                # Imported here (and only here) so the stub path -- and unit
                # tests in general -- never pull OpenVINO into the process.
                from src.engine.ov_genai.vlm import OVGenAI_VLM

                model = OVGenAI_VLM(load_config)
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

    async def _do_generate(self, msg: Dict[str, Any]) -> None:
        req_id = msg["req_id"]
        if self.model is None:
            await self.send(
                proto.encode_response(
                    proto.MSG_ERROR,
                    req_id=req_id,
                    error={"type": "NotLoaded", "message": "no model loaded"},
                )
            )
            return
        if self._active_gen is not None and not self._active_gen.done():
            await self.send(
                proto.encode_response(
                    proto.MSG_ERROR,
                    req_id=req_id,
                    error={"type": "Busy", "message": "a generation is already in progress"},
                )
            )
            return
        self._active_gen = asyncio.create_task(
            self._run_generation(msg), name=f"ovworker-gen-{req_id[:8]}"
        )

    async def _run_generation(self, msg: Dict[str, Any]) -> None:
        req_id = msg["req_id"]
        self._active_request_id = msg.get("request_id")
        try:
            raw = json.loads(msg["gen_config"])
            # The contract declares messages/input_ids as non-optional lists
            # with a None default (pydantic skips validation of defaults), so
            # a JSON round-trip must normalise them back to empty lists.
            raw["messages"] = raw.get("messages") or []
            raw["input_ids"] = raw.get("input_ids") or []
            gen_config = OVGenAI_GenConfig.model_validate(raw)
            async for item in self.model.generate_type(gen_config):
                await self.send(proto.encode_response(proto.MSG_ITEM, req_id=req_id, item=item))
            await self.send(proto.encode_response(proto.MSG_DONE, req_id=req_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[{self.model_name}] generation failed", exc_info=True)
            if proto.is_non_recoverable_error(e):
                # Wedged device: report FATAL and take the process down so
                # the supervisor can respawn with a fresh ov::Core.
                try:
                    await self.send(
                        proto.encode_response(proto.MSG_FATAL, error=proto.serialize_error(e))
                    )
                except Exception:
                    pass
                os._exit(1)
            await self.send(
                proto.encode_response(
                    proto.MSG_ERROR, req_id=req_id, error=proto.serialize_error(e)
                )
            )
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
