"""
End-to-end tests for the plain-OpenVINO out-of-process workers (stage 3):
supervisor + plain protocol + RemoteOV_Kokoro / RemoteOVQwen3ASR /
RemoteOVQwen3TTS facades, with the worker's models replaced by stubs
(OPENARC_WORKER_STUB=1) so no OpenVINO, GPU, or model files are needed.
Each test spawns a real child process and exercises the full pipe:
spawn -> LOAD -> RUN / RUN_STREAM -> ITEM / RESULT / DONE -> UNLOAD, plus
the failure paths (FATAL + respawn, native crash + respawn, recoverable
error) and the full ModelRegistry + WorkerRegistry route (including the
server-side WAV encoding and the streaming PCM path).
"""

import asyncio
import base64
from typing import Optional

import numpy as np
import pytest  # type: ignore[import]

from src.engine.worker import protocol as proto
from src.engine.worker.plain.worker_client import (
    RemoteOV_Kokoro,
    RemoteOVQwen3ASR,
    RemoteOVQwen3TTS,
)
from src.server.schemas.modeling.contract_kokoro import (
    OV_KokoroGenConfig,
    KokoroLanguage,
    KokoroVoice,
)
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.modeling.contract_qwen3tts import OV_Qwen3TTSCustomVoice
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType


def _load_config(
    tmp_path,
    name: str = "stub-kokoro",
    model_type: ModelType = ModelType.KOKORO,
) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(tmp_path),
        model_name=name,
        model_type=model_type,
        engine=EngineType.OPENVINO,
        device="CPU",
    )


async def _wait_for_state(facade, state: str, timeout: float = 30.0) -> None:
    async def _poll():
        while facade.status()["state"] != state:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


async def _wait_for_respawn(facade, first_pid: Optional[int], timeout: float = 30.0) -> None:
    """Wait until the supervisor detected the death and the respawned worker
    is ready again (a fresh PID proves a fresh process/pipeline)."""

    async def _poll():
        while facade.worker_pid == first_pid or facade.status()["state"] != "ready":
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout)


def _asr_config(payload: str) -> OV_Qwen3ASRGenConfig:
    return OV_Qwen3ASRGenConfig(
        audio_base64=base64.b64encode(payload.encode("utf-8")).decode("ascii")
    )


def _tts_config(text: str = "hello") -> OV_Qwen3TTSCustomVoice:
    return OV_Qwen3TTSCustomVoice(input=text, speaker="Cherry")


def _kokoro_config(text: str) -> OV_KokoroGenConfig:
    # All fields passed explicitly: pyright's pydantic plugin does not pick
    # up the positional Field() defaults in the contract, so a bare
    # OV_KokoroGenConfig(input=...) would be flagged as a call error.
    return OV_KokoroGenConfig(
        input=text,
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=400,
        response_format="wav",
    )


# --- facade-level tests -----------------------------------------------------------


def test_kokoro_load_stream_and_unload(tmp_path, monkeypatch) -> None:
    """Kokoro through the full pipe: RUN_STREAM yields one audio chunk per
    text chunk, and the server-side contract (torch tensors) is preserved."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path, name="stub-kokoro")

    async def _run():
        facade = RemoteOV_Kokoro(config, load_timeout=30.0)
        await facade.load_model(config)
        chunks = [c async for c in facade.chunk_forward_pass(_kokoro_config("hi"))]
        assert len(chunks) == 2
        import torch

        full = torch.cat([c.audio for c in chunks], dim=0)
        assert np.allclose(full.numpy(), [0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        assert chunks[0].chunk_text == "stub chunk 0"
        assert chunks[1].chunk_index == 1
        assert chunks[0].total_chunks == 2
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_asr_transcribe_contract(tmp_path, monkeypatch) -> None:
    """Qwen3-ASR through the full pipe: RUN -> RESULT -> (text, metrics,
    segments), the in-process return contract."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path, name="stub-asr", model_type=ModelType.QWEN3_ASR)

    async def _run():
        facade = RemoteOVQwen3ASR(config, load_timeout=30.0)
        await facade.load_model(config)
        text, metrics, segments = await facade.transcribe(_asr_config("some audio"))
        assert text == "stub asr transcript"
        assert metrics == {"num_chunks": 1}
        assert segments[0]["text"] == "stub asr transcript"
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_tts_generate_and_stream(tmp_path, monkeypatch) -> None:
    """Qwen3-TTS through the full pipe: non-streaming RUN -> (wav, sr) and
    streaming RUN_STREAM -> .audio chunks (async generator)."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(
        tmp_path, name="stub-tts", model_type=ModelType.QWEN3_TTS_CUSTOM_VOICE
    )

    async def _run():
        facade = RemoteOVQwen3TTS(config, load_timeout=30.0)
        await facade.load_model(config)

        wav, sr = await facade.generate(_tts_config())
        assert sr == 24000
        assert np.allclose(wav, [0.5, -0.5, 0.25, -0.25])

        chunks = [c async for c in facade.generate_stream(_tts_config())]
        assert len(chunks) == 2
        assert np.allclose(chunks[0].audio, [0.5, -0.5])
        assert np.allclose(chunks[1].audio, [0.5, -0.5])
        await facade._supervisor.unload()

    asyncio.run(_run())


# --- failure paths ------------------------------------------------------------------


def test_kokoro_fatal_error_respawns_worker(tmp_path, monkeypatch) -> None:
    """FATAL (non-recoverable) on a plain engine: the request fails with a
    dead-worker error and the supervisor respawns a fresh process."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path, name="stub-kokoro")

    async def _run():
        facade = RemoteOV_Kokoro(config, load_timeout=30.0)
        await facade.load_model(config)
        first_pid = facade.worker_pid

        with pytest.raises(proto.RemoteWorkerDeadError):
            _ = [
                c
                async for c in facade.chunk_forward_pass(
                    _kokoro_config("FATAL please")
                )
            ]
        await _wait_for_respawn(facade, first_pid)
        chunks = [
            c
            async for c in facade.chunk_forward_pass(_kokoro_config("still serving"))
        ]
        assert len(chunks) == 2
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_tts_crash_respawns_worker(tmp_path, monkeypatch) -> None:
    """A native crash (no FATAL message) on a plain engine: the request
    fails and the supervisor respawns -- the segfault-isolation story."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(
        tmp_path, name="stub-tts", model_type=ModelType.QWEN3_TTS_CUSTOM_VOICE
    )

    async def _run():
        facade = RemoteOVQwen3TTS(config, load_timeout=30.0)
        await facade.load_model(config)
        first_pid = facade.worker_pid

        with pytest.raises(proto.RemoteWorkerDeadError):
            await facade.generate(_tts_config("CRASH"))
        await _wait_for_respawn(facade, first_pid)
        wav, sr = await facade.generate(_tts_config())
        assert sr == 24000
        assert np.allclose(wav, [0.5, -0.5, 0.25, -0.25])
        await facade._supervisor.unload()

    asyncio.run(_run())


def test_asr_recoverable_error_keeps_worker_up(tmp_path, monkeypatch) -> None:
    """A recoverable per-request error on a plain engine: the request fails
    with a plain worker error and the worker stays up (no respawn)."""
    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    config = _load_config(tmp_path, name="stub-asr", model_type=ModelType.QWEN3_ASR)

    async def _run():
        facade = RemoteOVQwen3ASR(config, load_timeout=30.0)
        await facade.load_model(config)
        pid = facade.worker_pid

        with pytest.raises(proto.RemoteWorkerError) as exc:
            await facade.transcribe(_asr_config("RAISE"))
        assert "stub recoverable error" in str(exc.value)
        assert not isinstance(exc.value, proto.RemoteWorkerDeadError)

        # Same process, still serving.
        assert facade.worker_pid == pid
        assert facade.status()["state"] == "ready"
        text, _, _ = await facade.transcribe(_asr_config("fine audio"))
        assert text == "stub asr transcript"
        await facade._supervisor.unload()

    asyncio.run(_run())


# --- full registry route --------------------------------------------------------------


def test_plain_worker_registry_end_to_end(tmp_path, monkeypatch) -> None:
    """Full path for Kokoro + Qwen3-ASR + Qwen3-TTS: register_load -> facade
    -> worker process -> WorkerRegistry.generate_speech_kokoro /
    transcribe_qwen3_asr / generate_speech_qwen3_tts /
    stream_generate_speech_qwen3_tts, with WAV encoding on the server side
    and the worker-death policy (a FATAL must not unload the model)."""
    from src.server.model_registry import ModelRegistry
    from src.server.worker_registry import WorkerRegistry

    monkeypatch.setenv("OPENARC_WORKER_STUB", "1")
    kokoro_config = _load_config(tmp_path, name="stub-kokoro", model_type=ModelType.KOKORO)
    asr_config = _load_config(tmp_path, name="stub-asr", model_type=ModelType.QWEN3_ASR)
    tts_config = _load_config(
        tmp_path, name="stub-tts", model_type=ModelType.QWEN3_TTS_CUSTOM_VOICE
    )

    async def _run():
        registry = ModelRegistry()
        workers = WorkerRegistry(registry)
        kokoro_id = await registry.register_load(kokoro_config)
        asr_id = await registry.register_load(asr_config)
        tts_id = await registry.register_load(tts_config)

        async def _facade(model_id):
            async with registry._lock:
                for rec in registry._models.values():
                    if rec.model_id == model_id:
                        return rec.model_instance
            return None

        kokoro_facade = await _facade(kokoro_id)
        assert isinstance(kokoro_facade, RemoteOV_Kokoro)
        assert isinstance(await _facade(asr_id), RemoteOVQwen3ASR)
        assert isinstance(await _facade(tts_id), RemoteOVQwen3TTS)

        # Kokoro: speech out as a base64 WAV, encoded on the server side.
        result = await workers.generate_speech_kokoro(
            kokoro_config.model_name, _kokoro_config("hello")
        )
        wav_bytes = base64.b64decode(result["audio_base64"])
        assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
        assert result["metrics"]["chunks_processed"] == 2

        # Kokoro fatal error: the request fails, the model stays loaded, and
        # the supervisor respawns (the registry must not unload).
        kokoro_pid = kokoro_facade.worker_pid
        with pytest.raises(proto.RemoteWorkerDeadError):
            await workers.generate_speech_kokoro(
                kokoro_config.model_name, _kokoro_config("FATAL please")
            )
        await _wait_for_respawn(kokoro_facade, kokoro_pid)
        result = await workers.generate_speech_kokoro(
            kokoro_config.model_name, _kokoro_config("still serving")
        )
        assert base64.b64decode(result["audio_base64"])[:4] == b"RIFF"

        # Qwen3-ASR: transcription with segments.
        result = await workers.transcribe_qwen3_asr(
            asr_config.model_name, _asr_config("some audio")
        )
        assert result["text"] == "stub asr transcript"
        assert result["segments"][0]["text"] == "stub asr transcript"

        # Qwen3-TTS: speech out as a base64 WAV.
        result = await workers.generate_speech_qwen3_tts(
            tts_config.model_name, _tts_config()
        )
        wav_bytes = base64.b64decode(result["audio_base64"])
        assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
        assert result["metrics"]["sample_rate"] == 24000

        # Qwen3-TTS streaming: raw int16 PCM chunks on the stream.
        pcm_chunks = [
            c
            async for c in workers.stream_generate_speech_qwen3_tts(
                tts_config.model_name, _tts_config()
            )
        ]
        pcm = b"".join(pcm_chunks)
        # 2 stub chunks x 2 samples x 2 bytes (int16 LE) = 8 bytes.
        assert len(pcm) == 8

        # All three models survived; unload terminates all three processes.
        async with registry._lock:
            names = {r.model_name for r in registry._models.values()}
        assert {
            kokoro_config.model_name,
            asr_config.model_name,
            tts_config.model_name,
        } <= names
        assert await registry.register_unload(kokoro_config.model_name) is True
        assert await registry.register_unload(asr_config.model_name) is True
        assert await registry.register_unload(tts_config.model_name) is True
        asr_facade = await _facade(asr_id)
        tts_facade = await _facade(tts_id)
        assert asr_facade is not None and tts_facade is not None
        for _ in range(500):
            if all(
                f.status()["state"] == "closed"
                for f in (kokoro_facade, asr_facade, tts_facade)
            ):
                break
            await asyncio.sleep(0.01)
        assert kokoro_facade.status()["state"] == "closed"

    asyncio.run(_run())


# --- env switch ----------------------------------------------------------------------


def test_openvino_worker_env_switch(tmp_path, monkeypatch) -> None:
    """OPENARC_OPENVINO_WORKER gates the plain-OpenVINO workers (default on);
    it must not affect the GenAI switch and vice versa."""
    from src.server.model_registry import (
        _openvino_worker_enabled,
        _ovgenai_worker_enabled,
        _worker_enabled,
    )
    from src.server.schemas.registration import EngineType

    monkeypatch.delenv("OPENARC_OPENVINO_WORKER", raising=False)
    monkeypatch.delenv("OPENARC_OVGENAI_WORKER", raising=False)
    assert _openvino_worker_enabled(ModelType.KOKORO) is True
    assert _worker_enabled(EngineType.OPENVINO, ModelType.KOKORO) is True
    assert _worker_enabled(EngineType.OV_GENAI, ModelType.LLM) is True

    monkeypatch.setenv("OPENARC_OPENVINO_WORKER", "0")
    assert _openvino_worker_enabled(ModelType.KOKORO) is False
    assert _worker_enabled(EngineType.OPENVINO, ModelType.KOKORO) is False
    # The GenAI switch is independent.
    assert _ovgenai_worker_enabled(ModelType.LLM) is True

    monkeypatch.setenv("OPENARC_OVGENAI_WORKER", "off")
    monkeypatch.delenv("OPENARC_OPENVINO_WORKER", raising=False)
    assert _worker_enabled(EngineType.OV_GENAI, ModelType.LLM) is False
    assert _worker_enabled(EngineType.OPENVINO, ModelType.QWEN3_ASR) is True

    # Optimum models are not worker-routed (a later stage).
    assert _worker_enabled(EngineType.OV_OPTIMUM, ModelType.EMB) is False
