import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest  # type: ignore[import]

import src.server.routes.openai as openai_module
from src.server.models.registration import ModelType


_AUDIO_BYTES = b"audio-bytes"
_RESULT = {
    "text": "hello world",
    "metrics": {"language": "english", "audio_duration_sec": 4.0, "rtf": 0.5},
    "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "hello world"}],
}


class _FakeUpload:
    async def read(self) -> bytes:
        return _AUDIO_BYTES


def _call(monkeypatch: pytest.MonkeyPatch, response_format: str, result=None, openarc_asr=None):
    """Invoke the transcription handler with a loaded qwen3-asr model mocked in."""
    result = _RESULT if result is None else result

    fake_registry = SimpleNamespace(
        _lock=asyncio.Lock(),
        _models={
            "qwen3": SimpleNamespace(model_name="qwen3-asr", model_type=ModelType.QWEN3_ASR),
        },
    )
    transcribe_mock = AsyncMock(return_value=result)
    fake_workers = SimpleNamespace(transcribe_qwen3_asr=transcribe_mock)

    monkeypatch.setattr(openai_module, "_registry", fake_registry)
    monkeypatch.setattr(openai_module, "_workers", fake_workers)

    response = asyncio.run(
        openai_module.openai_audio_transcriptions(
            file=_FakeUpload(),
            model="qwen3-asr",
            response_format=response_format,
            openarc_asr=openarc_asr,
        )
    )
    return response, transcribe_mock


def test_json_returns_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _ = _call(monkeypatch, "json")
    assert response == {"text": "hello world"}


def test_verbose_json_includes_segments_and_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _ = _call(monkeypatch, "verbose_json")
    assert response == {
        "text": "hello world",
        "language": "english",
        "duration": 4.0,  # falls back to audio_duration_sec (metrics has no "duration")
        "segments": _RESULT["segments"],
        "metrics": _RESULT["metrics"],
    }


def test_diarized_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _ = _call(monkeypatch, "diarized_json")
    assert response == {
        "duration": 4.0,
        "segments": _RESULT["segments"],
        "task": "transcribe",
        "text": "hello world",
    }


def test_duration_prefers_explicit_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {
        "text": "hi",
        "metrics": {"duration": 9.9, "audio_duration_sec": 4.0},
        "segments": [],
    }
    response, _ = _call(monkeypatch, "verbose_json", result=result)
    assert response["duration"] == 9.9


def test_missing_segments_defaults_to_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {"text": "hi", "metrics": {"audio_duration_sec": 1.0}}  # no "segments" key
    response, _ = _call(monkeypatch, "verbose_json", result=result)
    assert response["segments"] == []


def test_defaults_used_when_openarc_asr_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # openarc_asr is None -> handler builds a default qwen3 gen_config and still
    # forwards the uploaded audio as base64.
    _, transcribe_mock = _call(monkeypatch, "json", openarc_asr=None)

    transcribe_mock.assert_awaited_once()
    _model_arg, gen_config = transcribe_mock.await_args.args
    assert gen_config.audio_base64 == base64.b64encode(_AUDIO_BYTES).decode("utf-8")
