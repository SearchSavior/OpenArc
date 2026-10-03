import base64

import pytest  # type: ignore[import]

from src.engine.audio import AudioDecodeError
from src.engine.openvino.qwen3_asr.qwen3_asr_utils import load_audio_any


def test_qwen3_load_audio_rejects_garbage_base64() -> None:
    with pytest.raises(AudioDecodeError, match="Unreadable audio"):
        load_audio_any(base64.b64encode(b"not audio" * 100).decode())


def test_decode_error_is_a_value_error() -> None:
    # routes map ValueError to HTTP 400
    assert issubclass(AudioDecodeError, ValueError)
