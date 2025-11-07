import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest  # type: ignore[import]

import src.engine.openvino.kokoro as kokoro_module
from src.engine.openvino.kokoro import OV_Kokoro
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.openvino import (
    KokoroLanguage,
    KokoroVoice,
    OV_KokoroGenConfig,
)


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO"


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-kokoro",
        model_type=ModelType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",
        runtime_config={"config": "value"},
    )


def test_make_chunks_respects_sentence_boundaries(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)

    text = "Hello world. This is a test! Another sentence?"
    chunks = kokoro.make_chunks(text, chunk_size=20)

    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "Hello world." in chunks[0]


def test_load_model_sets_model_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "kokoro"
    model_dir.mkdir()

    config = {
        "vocab": ["a", "b"],
        "plbert": {"max_position_embeddings": 256},
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "openvino_model.xml").write_text("<xml />", encoding="utf-8")

    core_instance = MagicMock()
    core_instance.compile_model.return_value = "compiled-model"
    monkeypatch.setattr(kokoro_module.ov, "Core", MagicMock(return_value=core_instance))

    load_config = ModelLoadConfig(
        model_path=str(model_dir),
        model_name="unit-kokoro",
        model_type=ModelType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",
        runtime_config={},
    )

    kokoro = OV_Kokoro(load_config)
    compiled = kokoro.load_model(load_config)

    core_instance.compile_model.assert_called_once_with(model_dir / "openvino_model.xml", "CPU")
    assert compiled == "compiled-model"
    assert kokoro.model == "compiled-model"
    assert kokoro.vocab == ["a", "b"]
    assert kokoro.context_length == 256


def test_chunk_forward_pass_yields_chunks(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro.model = object()
    kokoro.make_chunks = MagicMock(return_value=["Chunk one", "Chunk two"])  # type: ignore[assignment]

    async def immediate_to_thread(func, *args, **kwargs):  # type: ignore[override]
        return func(*args, **kwargs)

    monkeypatch.setattr(kokoro_module.asyncio, "to_thread", immediate_to_thread)

    pipeline_calls = []

    class DummyResult:
        def __init__(self, text: str) -> None:
            self.audio = f"audio:{text}"

    class DummyPipeline:
        def __init__(self, model, lang_code):
            pipeline_calls.append(("init", model, lang_code))

        def __call__(self, text, voice, speed):
            pipeline_calls.append(("call", text, voice, speed))
            yield DummyResult(text)

    monkeypatch.setattr(kokoro_module, "KPipeline", DummyPipeline)

    config = OV_KokoroGenConfig(
        kokoro_message="ignored",
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=50,
        response_format="wav",
    )

    async def _run_test():
        results = []
        async for item in kokoro.chunk_forward_pass(config):
            results.append(item)
        return results

    chunks = asyncio.run(_run_test())

    assert [chunk.chunk_text for chunk in chunks] == ["Chunk one", "Chunk two"]
    assert chunks[0].chunk_index == 0
    assert chunks[-1].total_chunks == 2
    assert pipeline_calls[0][0] == "init"
    assert pipeline_calls[1][0] == "call"


def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro.model = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(kokoro_module.gc, "collect", gc_mock)

    result = asyncio.run(kokoro.unload_model(registry, "model-name"))

    assert result is True
    assert kokoro.model is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

