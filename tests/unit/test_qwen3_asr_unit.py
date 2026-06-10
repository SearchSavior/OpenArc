import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest  # type: ignore[import]

import src.engine.openvino.qwen3_asr.qwen3_asr as qwen3_asr_module
from src.engine.openvino.qwen3_asr.qwen3_asr import OVQwen3ASR, SAMPLE_RATE
from src.server.models.openvino import OV_Qwen3ASRGenConfig
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.engine.openvino.qwen3_asr.qwen3_asr_utils import (
    LANGUAGE_CODE_TO_NAME,
    SUPPORTED_LANGUAGES,
    resolve_language_name,
    validate_language,
)

# audio_chunks returns (raw_output, chunk_metrics); collect_metrics sums these keys.
_CHUNK_METRICS = {
    "feature_sec": 1.0,
    "encoder_sec": 1.0,
    "prefill_sec": 1.0,
    "decode_sec": 1.0,
    "detok_sec": 1.0,
    "prompt_tokens": 10,
    "generated_tokens": 20,
    "encoder_tokens": 5,
}


def _make_asr() -> OVQwen3ASR:
    """Build an OVQwen3ASR without running __init__ (which reads model files)."""
    asr = OVQwen3ASR.__new__(OVQwen3ASR)
    asr.load_config = ModelLoadConfig(
        model_path="unused",
        model_name="test-qwen3-asr",
        model_type=ModelType.QWEN3_ASR,
        engine=EngineType.OPENVINO,
        device="CPU",
    )
    asr.t_model_load = 0.0
    return asr


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_array: np.ndarray,
    chunk_items: list,
    parse_outputs: list,
) -> OVQwen3ASR:
    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(qwen3_asr_module.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(qwen3_asr_module, "normalize_audios", lambda _audio_input: [audio_array])
    monkeypatch.setattr(
        qwen3_asr_module,
        "split_audio_into_chunks",
        lambda **kwargs: chunk_items,
    )
    monkeypatch.setattr(
        qwen3_asr_module,
        "parse_asr_output",
        MagicMock(side_effect=parse_outputs),
    )
    monkeypatch.setattr(qwen3_asr_module, "merge_languages", lambda _langs: "english")

    asr = _make_asr()
    asr.audio_chunks = MagicMock(return_value=("raw", dict(_CHUNK_METRICS)))
    return asr


def test_transcribe_returns_text_metrics_and_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_items = [
        (np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32), 0.0),
        (np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32), 1.5),
    ]
    asr = _patch_pipeline(
        monkeypatch,
        audio_array=np.zeros(SAMPLE_RATE * 4, dtype=np.float32),  # 4.0s
        chunk_items=chunk_items,
        parse_outputs=[("en", "hello "), ("en", "world")],
    )

    text, metrics, segments = asyncio.run(
        asr.transcribe(OV_Qwen3ASRGenConfig(audio_base64="AAA"))
    )

    assert text == "hello world"
    assert segments == [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "hello "},
        {"id": 1, "start": 1.5, "end": 3.5, "text": "world"},
    ]
    # Metrics aggregate across the two chunks.
    assert metrics["language"] == "english"
    assert metrics["audio_duration_sec"] == 4.0
    assert metrics["feature_sec"] == 2.0
    assert metrics["prompt_tokens"] == 20
    assert metrics["generated_tokens"] == 40
    assert metrics["encoder_tokens"] == 10
    # collect_metrics derives throughput from the summed values.
    assert metrics["prefill_tok_s"] == 20 / 2.0
    assert metrics["decode_tok_s"] == 40 / 2.0


def test_transcribe_skips_empty_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Middle chunk yields no text -> no segment appended for it.
    chunk_items = [
        (np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), 0.0),
        (np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), 1.0),
        (np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), 2.0),
    ]
    asr = _patch_pipeline(
        monkeypatch,
        audio_array=np.zeros(SAMPLE_RATE * 3, dtype=np.float32),
        chunk_items=chunk_items,
        parse_outputs=[("en", "first"), ("en", ""), ("en", "third")],
    )

    text, _metrics, segments = asyncio.run(
        asr.transcribe(OV_Qwen3ASRGenConfig(audio_base64="AAA"))
    )

    assert text == "firstthird"
    assert len(segments) == 2
    # NOTE: documents current behavior -- `id` is the chunk index, so skipping a
    # middle chunk produces non-contiguous ids (0, 2) rather than OpenAI's
    # contiguous segment indexing (0, 1).
    assert [seg["id"] for seg in segments] == [0, 2]
    assert [seg["text"] for seg in segments] == ["first", "third"]


def test_transcribe_empty_audio_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    asr = _patch_pipeline(
        monkeypatch,
        audio_array=np.zeros(0, dtype=np.float32),  # 0s -> early return
        chunk_items=[],
        parse_outputs=[],
    )

    result = asyncio.run(asr.transcribe(OV_Qwen3ASRGenConfig(audio_base64="AAA")))

    assert result == ("", {}, [])

def test_resolve_language_name_maps_iso_639_1_code() -> None:
    assert resolve_language_name("en") == "English"
    assert resolve_language_name("zh") == "Chinese"


def test_resolve_language_name_code_is_case_insensitive_and_trimmed() -> None:
    assert resolve_language_name("ZH") == "Chinese"
    assert resolve_language_name(" ja ") == "Japanese"


def test_resolve_language_name_passes_through_full_names() -> None:
    assert resolve_language_name("english") == "English"
    assert resolve_language_name("cHINese") == "Chinese"


def test_resolve_language_name_handles_non_iso_639_1_special_cases() -> None:
    # Cantonese has no ISO-639-1 code; Filipino accepts both 639-1 (tl) and 639-3 (fil).
    assert resolve_language_name("yue") == "Cantonese"
    assert resolve_language_name("tl") == "Filipino"
    assert resolve_language_name("fil") == "Filipino"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_resolve_language_name_rejects_empty(value) -> None:
    with pytest.raises(ValueError):
        resolve_language_name(value)


def test_every_mapped_code_resolves_to_a_supported_language() -> None:
    # Guards against drift if SUPPORTED_LANGUAGES is edited without updating the map.
    for code, name in LANGUAGE_CODE_TO_NAME.items():
        assert name in SUPPORTED_LANGUAGES, f"{code} -> {name} not in SUPPORTED_LANGUAGES"
        # Resolved canonical names must pass validation.
        validate_language(resolve_language_name(code))
