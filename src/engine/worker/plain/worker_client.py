"""
Remote plain-OpenVINO facades (stage 3) -- in-process stand-ins for the
Kokoro / Qwen3-ASR / Qwen3-TTS engines that run in a supervised worker
subprocess, for segfault isolation: a native crash in OpenVINO inference
takes down the worker process, which the supervisor respawns with a fresh
pipeline, while the server process keeps serving.

Each facade presents the same surface the server calls on the corresponding
in-process class:

  * RemoteOV_Kokoro.chunk_forward_pass  -> async gen of .audio chunks
                                           (same as OV_Kokoro)
  * RemoteOVQwen3ASR.transcribe         -> (text, metrics, segments)
                                           (same as OVQwen3ASR)
  * RemoteOVQwen3TTS.generate           -> (wav float32 array, sample_rate)
                                           (same as OVQwen3TTS)
  * RemoteOVQwen3TTS.generate_stream    -> async gen of .audio chunks
                                           (the in-process one is a sync
                                           generator; the registry's
                                           streaming path handles both)

The worker does the inference (the segfault-prone part) and returns audio as
base64 float32 sample arrays; the server keeps the WAV encoding, so the
public response shapes are unchanged.
"""

from __future__ import annotations

import base64
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Tuple

import numpy as np

from src.engine.worker import protocol as proto  # noqa: F401  (re-exported error types)
from src.engine.worker.plain import protocol as p_proto
from src.engine.worker.plain.supervisor import PlainWorkerSupervisor
from src.engine.worker.worker_client import RemoteEngine
from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.modeling.contract_qwen3tts import OV_Qwen3TTSGenConfig
from src.server.schemas.registration import ModelLoadConfig

if TYPE_CHECKING:
    from src.server.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def _decode_audio_b64(item: Dict[str, Any]) -> np.ndarray:
    """Decode a protocol audio item (base64 float32 samples) into a 1-D array."""
    return np.frombuffer(base64.b64decode(item["audio_b64"]), dtype=np.float32)


class RemoteOV_Kokoro(RemoteEngine):
    """Out-of-process Kokoro TTS (stage 3)."""

    SUPERVISOR_CLS = PlainWorkerSupervisor

    def chunk_forward_pass(self, gen_config: OV_KokoroGenConfig) -> AsyncIterator[Any]:
        """Same yield contract as OV_Kokoro.chunk_forward_pass: objects with
        .audio (torch tensor), .chunk_text, .chunk_index, .total_chunks."""
        return self._kokoro_stream(gen_config)

    async def _kokoro_stream(self, gen_config: OV_KokoroGenConfig) -> AsyncIterator[Any]:
        # torch is imported lazily: the server process has it, but the worker
        # package must stay importable without it (worker children, tests).
        import torch

        async for item in self._run_stream(
            p_proto.OP_RUN_STREAM, gen_config.model_dump_json()
        ):
            yield SimpleNamespace(
                audio=torch.from_numpy(_decode_audio_b64(item)),
                chunk_text=item.get("chunk_text", ""),
                chunk_index=item.get("chunk_index", 0),
                total_chunks=item.get("total_chunks", 0),
            )


class RemoteOVQwen3ASR(RemoteEngine):
    """Out-of-process Qwen3-ASR (stage 3)."""

    SUPERVISOR_CLS = PlainWorkerSupervisor

    async def transcribe(
        self, gen_config: OV_Qwen3ASRGenConfig
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Same return contract as OVQwen3ASR.transcribe:
        (text, metrics, segments)."""
        result = await self._run_single(p_proto.OP_RUN, gen_config.model_dump_json())
        return (
            result.get("text", ""),
            result.get("metrics") or {},
            result.get("segments") or [],
        )


class RemoteOVQwen3TTS(RemoteEngine):
    """Out-of-process Qwen3-TTS (stage 3; all three TTS modes)."""

    SUPERVISOR_CLS = PlainWorkerSupervisor

    async def generate(self, gen_config: OV_Qwen3TTSGenConfig) -> Tuple[np.ndarray, int]:
        """Same return contract as OVQwen3TTS.generate:
        (wav float32 array, sample_rate)."""
        result = await self._run_single(p_proto.OP_RUN, gen_config.model_dump_json())
        return _decode_audio_b64(result), int(result.get("sample_rate", 24000))

    def generate_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> AsyncIterator[Any]:
        """Stream TTS audio chunks; each item has .audio (float32 array).

        Unlike the in-process OVQwen3TTS.generate_stream (a sync generator),
        this is an ASYNC generator -- the worker registry's streaming path
        handles both shapes.
        """
        return self._tts_stream(gen_config)

    async def _tts_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> AsyncIterator[Any]:
        async for item in self._run_stream(
            p_proto.OP_RUN_STREAM, gen_config.model_dump_json()
        ):
            yield SimpleNamespace(audio=_decode_audio_b64(item))
