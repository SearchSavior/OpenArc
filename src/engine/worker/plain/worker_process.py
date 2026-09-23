"""
Entry point of a plain-OpenVINO inference worker process (stage 3).

One of these runs per loaded plain-OpenVINO model (Kokoro TTS, Qwen3-ASR,
Qwen3-TTS). Unlike the GenAI workers (which stream text tokens), these
engines produce audio or transcription results, so the protocol adds
single-result (RUN -> RESULT) and audio-streaming (RUN_STREAM ->
ITEM* DONE) runs -- see src.engine.worker.plain.protocol.

The process boundary here is for SEGFAULT ISOLATION: a native crash in
OpenVINO inference (or in the audio toolchain) takes down only this process;
the supervisor respawns a fresh one with a fresh pipeline, and the server
process keeps serving other models.

The child inherits the whole protocol loop (stdin reader, load flow,
gen-config validation, FATAL/ERROR ladder, shutdown) from the GenAI worker
(src.engine.worker.worker_process._Worker) and only overrides model
construction, gen-config contracts, and the run ops.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Optional

import numpy as np

from src.engine.worker import protocol as proto
from src.engine.worker.plain import protocol as p_proto
from src.engine.worker.worker_process import (
    _STUB_ON,
    _Worker,
    _configure_logging,
)
from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.modeling.contract_qwen3tts import (
    OV_Qwen3TTSCustomVoice,
    OV_Qwen3TTSGenConfig,
    OV_Qwen3TTSVoiceClone,
    OV_Qwen3TTSVoiceDesign,
)
from src.server.schemas.registration import ModelLoadConfig, ModelType

logger = logging.getLogger("openarc.worker.plain")

# Engines this worker process knows how to run (stage 3). Everything else
# stays in the server process (or, for GenAI, in a GenAI worker).
_WORKER_MODEL_TYPES = (
    ModelType.KOKORO,
    ModelType.QWEN3_ASR,
    ModelType.QWEN3_TTS_CUSTOM_VOICE,
    ModelType.QWEN3_TTS_VOICE_DESIGN,
    ModelType.QWEN3_TTS_VOICE_CLONE,
)

_TTS_TYPES = (
    ModelType.QWEN3_TTS_CUSTOM_VOICE,
    ModelType.QWEN3_TTS_VOICE_DESIGN,
    ModelType.QWEN3_TTS_VOICE_CLONE,
)

_GEN_CONFIG_CLASSES = {
    ModelType.KOKORO: OV_KokoroGenConfig,
    ModelType.QWEN3_ASR: OV_Qwen3ASRGenConfig,
    ModelType.QWEN3_TTS_CUSTOM_VOICE: OV_Qwen3TTSCustomVoice,
    ModelType.QWEN3_TTS_VOICE_DESIGN: OV_Qwen3TTSVoiceDesign,
    ModelType.QWEN3_TTS_VOICE_CLONE: OV_Qwen3TTSVoiceClone,
}


def _make_model(load_config: ModelLoadConfig) -> Any:
    """Build the model object inside the worker process (real or stub)."""
    if load_config.model_type not in _WORKER_MODEL_TYPES:
        raise ValueError(
            f"model type {load_config.model_type.value!r} is not supported "
            f"in the plain-OpenVINO worker process (supported: "
            f"{', '.join(t.value for t in _WORKER_MODEL_TYPES)})"
        )
    if os.environ.get("OPENARC_WORKER_STUB", "").strip().lower() in _STUB_ON:
        return _PlainStubModel(load_config)
    # Imported here (and only here) so the stub path -- and unit tests in
    # general -- never pull OpenVINO/torch into the process.
    if load_config.model_type == ModelType.KOKORO:
        from src.engine.openvino.kokoro import OV_Kokoro

        return OV_Kokoro(load_config)
    if load_config.model_type == ModelType.QWEN3_ASR:
        from src.engine.openvino.qwen3_asr.qwen3_asr import OVQwen3ASR

        return OVQwen3ASR(load_config)
    from src.engine.openvino.qwen3_tts.qwen3_tts import OVQwen3TTS

    return OVQwen3TTS(load_config)


# --- audio (de)serialization -------------------------------------------------------
# Audio crosses the pipe as base64-encoded 1-D float32 sample arrays. The
# inference (and any native audio processing) stays in this process; the
# server does the final WAV encoding, so its response shapes are unchanged.


def _to_float32_b64(audio: Any) -> tuple:
    """Encode 1-D audio (torch tensor or numpy array) as (base64, samples)."""
    if hasattr(audio, "detach"):  # torch tensor
        arr = audio.detach().cpu().numpy()
    else:
        arr = audio
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return base64.b64encode(arr.tobytes()).decode("ascii"), int(arr.size)


# --- per-engine run routines ---------------------------------------------------------


async def _kokoro_stream(model: Any, gen_config: Any) -> AsyncIterator[Dict[str, Any]]:
    """One protocol item per Kokoro text chunk: audio + chunk metadata."""
    async for chunk in model.chunk_forward_pass(gen_config):
        audio_b64, samples = _to_float32_b64(chunk.audio)
        yield {
            "audio_b64": audio_b64,
            "samples": samples,
            "chunk_text": getattr(chunk, "chunk_text", ""),
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "total_chunks": getattr(chunk, "total_chunks", 0),
        }


async def _asr_single(model: Any, gen_config: Any) -> Dict[str, Any]:
    text, metrics, segments = await model.transcribe(gen_config)
    return {"text": text, "metrics": metrics, "segments": segments}


async def _tts_single(model: Any, gen_config: Any) -> Dict[str, Any]:
    wav, sr = await model.generate(gen_config)
    audio_b64, samples = _to_float32_b64(wav)
    return {"audio_b64": audio_b64, "samples": samples, "sample_rate": int(sr)}


async def _tts_stream(model: Any, gen_config: Any) -> AsyncIterator[Dict[str, Any]]:
    """Stream Qwen3-TTS chunks; one protocol item per audio chunk.

    generate_stream is a blocking sync generator (the engine runs OpenVINO
    inference inside it), so advance it one chunk at a time in a worker
    thread; the event loop stays free between chunks.
    """
    gen = model.generate_stream(gen_config)
    sentinel = object()
    while True:
        chunk: Any = await asyncio.to_thread(next, gen, sentinel)
        if chunk is sentinel:
            break
        audio_b64, samples = _to_float32_b64(chunk.audio)
        yield {"audio_b64": audio_b64, "samples": samples}


# --- the worker -----------------------------------------------------------------------


class _PlainWorker(_Worker):
    """Plain-OpenVINO protocol loop (stage 3).

    Inherits from the GenAI _Worker: the stdin read loop, load flow,
    gen-config validation, FATAL/ERROR ladder, cancel and shutdown are all
    shared. This class only adds the model construction, the gen-config
    contracts, and the RUN / RUN_STREAM ops.
    """

    def _build_model(self, load_config: ModelLoadConfig) -> Any:
        return _make_model(load_config)

    def _gen_config_cls(self) -> Optional[Any]:
        if self.model_type is None:
            return None
        return _GEN_CONFIG_CLASSES.get(self.model_type)

    # -- RUN / RUN_STREAM dispatch -------------------------------------------------
    async def _handle_custom_op(self, msg: Dict[str, Any]) -> None:
        op = msg.get("op")
        if op == p_proto.OP_RUN:
            await self._start_single(msg)
        elif op == p_proto.OP_RUN_STREAM:
            await self._start_stream(msg)
        else:
            await super()._handle_custom_op(msg)

    async def _start_single(self, msg: Dict[str, Any]) -> None:
        err = await self._check_run_preconditions(msg)
        if err is not None:
            await self.send(err)
            return
        self._active_gen = asyncio.create_task(
            self._run_single_op(msg), name=f"ovworker-run-{msg['req_id'][:8]}"
        )

    async def _start_stream(self, msg: Dict[str, Any]) -> None:
        err = await self._check_run_preconditions(msg)
        if err is not None:
            await self.send(err)
            return
        self._active_gen = asyncio.create_task(
            self._run_stream_op(msg), name=f"ovworker-stream-{msg['req_id'][:8]}"
        )

    # -- run bodies ------------------------------------------------------------------
    def _single_result(self, model: Any, gen_config: Any) -> Any:
        """The coroutine producing the single result for the loaded engine."""
        if self.model_type == ModelType.QWEN3_ASR:
            return _asr_single(model, gen_config)
        if self.model_type in _TTS_TYPES:
            return _tts_single(model, gen_config)
        raise ValueError(f"no single-result run for model type {self.model_type!r}")

    def _stream_items(self, model: Any, gen_config: Any) -> AsyncIterator[Dict[str, Any]]:
        """The async iterator producing the audio items for the loaded engine."""
        if self.model_type == ModelType.KOKORO:
            return _kokoro_stream(model, gen_config)
        if self.model_type in _TTS_TYPES:
            return _tts_stream(model, gen_config)
        raise ValueError(f"no streaming run for model type {self.model_type!r}")

    async def _run_single_op(self, msg: Dict[str, Any]) -> None:
        req_id = msg["req_id"]
        self._active_request_id = msg.get("request_id")
        try:
            gen_config = self._validate_gen_config(json.loads(msg["gen_config"]))
            result = await self._single_result(self.model, gen_config)
            await self.send(
                p_proto.encode_response(p_proto.MSG_RESULT, req_id=req_id, result=result)
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._report_run_error(e, req_id, "run")
        finally:
            self._active_request_id = None
            self._active_gen = None

    async def _run_stream_op(self, msg: Dict[str, Any]) -> None:
        req_id = msg["req_id"]
        self._active_request_id = msg.get("request_id")
        try:
            gen_config = self._validate_gen_config(json.loads(msg["gen_config"]))
            async for item in self._stream_items(self.model, gen_config):
                await self.send(p_proto.encode_response(p_proto.MSG_ITEM, req_id=req_id, item=item))
            await self.send(p_proto.encode_response(p_proto.MSG_DONE, req_id=req_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._report_run_error(e, req_id, "run_stream")
        finally:
            self._active_request_id = None
            self._active_gen = None


# --- stub -----------------------------------------------------------------------------


class _PlainStubModel:
    """
    Test double for the plain-OpenVINO engines (Kokoro/ASR/TTS), selected by
    OPENARC_WORKER_STUB=1 so the whole spawn/IPC/respawn machinery can be
    exercised in unit tests without OpenVINO, a GPU, or model files. It
    mirrors the engine contracts:

      Kokoro:    chunk_forward_pass async-yields objects with .audio
                 (float32 array) + chunk_text/chunk_index/total_chunks
      Qwen3-ASR: transcribe -> (text, metrics, segments)
      Qwen3-TTS: generate -> (wav float32 array, sample_rate);
                 generate_stream sync-yields objects with .audio

    Control text (the config `input`, or the base64-decoded audio payload for
    ASR) drives behaviour with the same keywords as the GenAI stub:
      CRASH -> os._exit(139)
      FATAL -> raises a CL_OUT_OF_RESOURCES-looking error (non-recoverable)
      RAISE -> recoverable error (mid-stream for the streaming methods)
    """

    _STUB_SR = 24000

    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self.model_type = load_config.model_type

    def load_model(self, loader: ModelLoadConfig) -> None:
        if os.environ.get("OPENARC_WORKER_STUB_FAIL_LOAD", "").strip().lower() in _STUB_ON:
            raise RuntimeError("stub load failure")
        delay = float(os.environ.get("OPENARC_WORKER_STUB_LOAD_DELAY", "0.05"))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _control_text(gen_config: Any) -> str:
        """The control text: the config `input` for Kokoro/TTS, the
        (text-encoded) base64 audio payload for ASR."""
        audio = getattr(gen_config, "audio_base64", None)
        if audio:
            try:
                return base64.b64decode(audio).decode("utf-8", "replace")
            except Exception:
                return ""
        return getattr(gen_config, "input", "") or ""

    def _check_fatal(self, text: str) -> None:
        if "CRASH" in text:
            os._exit(139)
        if "FATAL" in text:
            raise RuntimeError("CL_OUT_OF_RESOURCES (stub)")

    # -- Kokoro ---------------------------------------------------------------
    async def chunk_forward_pass(self, gen_config: OV_KokoroGenConfig) -> Any:
        text = self._control_text(gen_config)
        self._check_fatal(text)
        for i in range(2):
            yield SimpleNamespace(
                audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                chunk_text=f"stub chunk {i}",
                chunk_index=i,
                total_chunks=2,
            )
            if i == 0 and "RAISE" in text:
                raise RuntimeError("stub recoverable error")

    # -- Qwen3-ASR -------------------------------------------------------------
    async def transcribe(self, gen_config: OV_Qwen3ASRGenConfig) -> Any:
        text = self._control_text(gen_config)
        self._check_fatal(text)
        if "RAISE" in text:
            raise RuntimeError("stub recoverable error")
        return (
            "stub asr transcript",
            {"num_chunks": 1},
            [{"id": 0, "start": 0.0, "end": 1.0, "text": "stub asr transcript"}],
        )

    # -- Qwen3-TTS ---------------------------------------------------------------
    async def generate(self, gen_config: OV_Qwen3TTSGenConfig) -> Any:
        text = self._control_text(gen_config)
        self._check_fatal(text)
        if "RAISE" in text:
            raise RuntimeError("stub recoverable error")
        return np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32), self._STUB_SR

    def generate_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> Any:
        text = self._control_text(gen_config)
        self._check_fatal(text)
        for i in range(2):
            yield SimpleNamespace(audio=np.array([0.5, -0.5], dtype=np.float32))
            if i == 0 and "RAISE" in text:
                raise RuntimeError("stub recoverable error")


def main() -> int:
    """Run the worker protocol loop. Returns the process exit code."""
    _configure_logging()
    worker = _PlainWorker()
    try:
        asyncio.run(worker.run())
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        logger.error(f"plain worker main loop crashed: {e}", exc_info=True)
        return 1
