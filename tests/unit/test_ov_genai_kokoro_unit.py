import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest  # type: ignore[import]
import torch

import src.engine.openvino.kokoro as kokoro_module
from src.engine.openvino.kokoro import OV_Kokoro
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType
from src.server.schemas.modeling.contract_kokoro import (
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


def test_load_model_forwards_runtime_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "kokoro"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"vocab": ["a"], "plbert": {"max_position_embeddings": 256}}),
        encoding="utf-8",
    )
    (model_dir / "openvino_model.xml").write_text("<xml />", encoding="utf-8")

    core_instance = MagicMock()
    core_instance.compile_model.return_value = "compiled-model"
    monkeypatch.setattr(kokoro_module.ov, "Core", MagicMock(return_value=core_instance))

    load_config = ModelLoadConfig(
        model_path=str(model_dir),
        model_name="rtc-kokoro",
        model_type=ModelType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",
        runtime_config={"NUM_STREAMS": "2"},
    )

    OV_Kokoro(load_config).load_model(load_config)

    core_instance.set_property.assert_called_once_with({"NUM_STREAMS": "2"})


def test_forward_runs_compiled_model(load_config: ModelLoadConfig) -> None:
    # KPipeline calls model(phonemes, ref_s, speed). That has to end up in the
    # compiled OpenVINO model, not KModel's PyTorch forward on CPU.
    kokoro = OV_Kokoro(load_config)
    kokoro.vocab = {"a": 5, "b": 7}
    kokoro.context_length = 512
    audio = np.linspace(-1, 1, 2400, dtype=np.float32)
    pred_dur = np.array([1, 2, 3, 1], dtype=np.int64)
    kokoro.model = MagicMock(return_value=(audio, pred_dur))
    ref_s = torch.zeros(1, 256)

    out = kokoro("ab", ref_s, speed=1.5, return_output=True)

    kokoro.model.assert_called_once()
    input_ids, passed_ref_s, speed = kokoro.model.call_args.args[0]
    assert input_ids.tolist() == [[0, 5, 7, 0]]
    assert passed_ref_s is ref_s
    assert speed.item() == 1.5
    assert torch.equal(out.audio, torch.from_numpy(audio))
    assert torch.equal(out.pred_dur, torch.from_numpy(pred_dur))


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

    monkeypatch.setattr("kokoro.pipeline.KPipeline", DummyPipeline)

    config = OV_KokoroGenConfig(
        input="ignored",
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


def test_make_chunks_splits_overlong_sentence_after_flush(load_config: ModelLoadConfig) -> None:
    """Regression: an oversized sentence arriving right after a buffer flush
    must still be split, not passed through whole."""
    kokoro = OV_Kokoro(load_config)

    short = "Short one."
    long_sentence = "word " * 30 + "end"  # 154 chars, no punctuation
    text = f"{short} {long_sentence}"
    chunks = kokoro.make_chunks(text, chunk_size=50)

    assert all(len(chunk) <= 50 for chunk in chunks)
    joined = " ".join(chunks)
    assert "end" in joined
    assert joined.split().count("word") == 30


def test_make_chunks_size_and_lossless_invariants(load_config: ModelLoadConfig) -> None:
    """Every chunk respects the size limit and no word is ever dropped."""
    kokoro = OV_Kokoro(load_config)

    text = (
        "First sentence here! Followed by a question? And one more, with a "
        "clause, to split on; plus a colon: like this. " * 8
    )
    chunks = kokoro.make_chunks(text, chunk_size=120)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_make_chunks_prefers_clause_boundary(load_config: ModelLoadConfig) -> None:
    """Mid-sentence splits should land on clause punctuation when available."""
    kokoro = OV_Kokoro(load_config)

    text = "before the comma there are words, after it there are more words and yet more and more words"
    chunks = kokoro.make_chunks(text, chunk_size=40)

    assert len(chunks) >= 2
    assert chunks[0].endswith(",")
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_make_chunks_handles_newlines_and_empty(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)

    assert kokoro.make_chunks("", 100) == []
    assert kokoro.make_chunks("   \n  ", 100) == []

    chunks = kokoro.make_chunks("Para one is short.\n\nPara two is also short.", 100)
    assert len(chunks) == 2


def test_chunk_forward_pass_concatenates_multi_bucket_results(
    monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig
) -> None:
    """Regression: when KPipeline yields multiple buckets for one text chunk,
    all audio must be concatenated — taking only the first drops speech."""
    import torch

    kokoro = OV_Kokoro(load_config)
    kokoro.model = object()
    kokoro.make_chunks = MagicMock(return_value=["Only chunk"])  # type: ignore[assignment]

    async def immediate_to_thread(func, *args, **kwargs):  # type: ignore[override]
        return func(*args, **kwargs)

    monkeypatch.setattr(kokoro_module.asyncio, "to_thread", immediate_to_thread)

    class DummyResult:
        def __init__(self, audio) -> None:
            self.audio = audio

    class DummyPipeline:
        def __init__(self, model, lang_code):
            pass

        def __call__(self, text, voice, speed):
            yield DummyResult(torch.zeros(10))
            yield DummyResult(torch.ones(10))

    monkeypatch.setattr("kokoro.pipeline.KPipeline", DummyPipeline)

    config = OV_KokoroGenConfig(
        input="ignored",
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=50,
        response_format="wav",
    )

    async def _run_test():
        return [item async for item in kokoro.chunk_forward_pass(config)]

    chunks = asyncio.run(_run_test())

    assert len(chunks) == 1
    assert chunks[0].audio.shape[0] == 20
    assert int(chunks[0].audio[:10].sum()) == 0
    assert int(chunks[0].audio[10:].sum()) == 10


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



def test_pad_reflect_mirrors_and_keeps_prefix() -> None:
    x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4)  # [1, 2, 3, 4]

    padded = kokoro_module._pad_reflect(x, 11)

    assert padded.shape[-1] == 11
    assert padded[..., :4].tolist() == x.tolist()
    assert padded.flatten().tolist() == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3]
    assert torch.equal(kokoro_module._pad_reflect(x, 4), x)


def test_gpu_path_pads_to_bucket_and_trims(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro._style_dim = 128
    kokoro._buckets = (64, 128)
    frames = 70
    calls = []

    def fake_decoder(inputs):
        calls.append([a.shape for a in inputs])
        return (np.ones((1, 1, inputs[0].shape[-1] * kokoro_module.SAMPLES_PER_FRAME), dtype=np.float32),)

    kokoro._gpu_decoders = {64: None, 128: fake_decoder}
    pred_dur = torch.tensor([3, 4])
    kokoro._front = MagicMock(return_value=(  # type: ignore[assignment]
        torch.randn(1, 512, frames), torch.randn(1, 2 * frames), torch.randn(1, 2 * frames), torch.zeros(1, 128), pred_dur,
    ))

    audio, dur = kokoro.forward_with_tokens(torch.tensor([[0, 5, 0]]), torch.zeros(1, 256), 1.0)

    assert calls == [[(1, 512, 128), (1, 256), (1, 256), (1, 128)]]
    assert audio.shape == (frames * kokoro_module.SAMPLES_PER_FRAME,)
    assert dur is pred_dur


def test_plan_cuts_prefers_pause_after_punctuation(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro._buckets = (100,)
    kokoro._space_id, kokoro._clause_end_ids = 16, {4}
    #                    BOS   a    .   sp    b   sp    c   EOS
    ids = torch.tensor([[0,    1,   4,  16,   2,  16,   3,  0]])
    dur = torch.tensor([10,   40,   5,  10,  30,  10,  40,  5])   # ends 10 50 55 65 95 105 145 150

    cuts = kokoro._plan_cuts(ids, dur)

    # the word gap at 100 is later, but the clause pause at 60 is in the back 60% of the window
    assert cuts == [60]


def test_plan_cuts_falls_back_to_word_gap_then_hard_cut(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro._buckets = (100,)
    kokoro._space_id, kokoro._clause_end_ids = 16, set()
    ids = torch.tensor([[0, 1, 16, 2, 3, 0]])
    dur = torch.tensor([5, 60, 10, 100, 100, 5])          # ends 5 65 75 175 275 280

    assert kokoro._plan_cuts(ids, dur) == [70, 170, 270]  # word gap, then no space in reach: hard cuts


def test_gpu_path_decodes_long_inputs_in_pieces(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    kokoro._buckets = (64,)
    kokoro._space_id, kokoro._clause_end_ids = 16, set()
    seen = []

    def fake_decoder(inputs):
        seen.append(inputs[0].shape[-1])
        return (np.zeros((1, 1, inputs[0].shape[-1] * 600), np.float32),)

    kokoro._gpu_decoders = {64: fake_decoder}
    dur = torch.tensor([10, 40, 10, 40, 10])               # 110 frames; the space's middle is frame 55
    kokoro._front = MagicMock(return_value=(  # type: ignore[assignment]
        torch.zeros(1, 512, 110), torch.zeros(1, 220), torch.zeros(1, 220), torch.zeros(1, 128), dur,
    ))

    audio, out_dur = kokoro.forward_with_tokens(torch.tensor([[0, 1, 16, 2, 0]]), torch.zeros(1, 256), 1.0)

    assert seen == [64, 64]                                # 55 + 55 frames, each padded to 64
    assert audio.numel() == 110 * 600                      # exact length, nothing inserted
    assert out_dur is dur


def test_polyphase_conv_transpose_is_exact() -> None:
    torch.manual_seed(0)
    cases = [
        dict(in_channels=8, out_channels=6, kernel_size=20, stride=10, padding=5),
        dict(in_channels=6, out_channels=4, kernel_size=12, stride=6, padding=3),
        dict(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1, groups=10),
        dict(in_channels=4, out_channels=4, kernel_size=5, stride=3, padding=0, bias=False),
    ]
    for case in cases:
        ct = torch.nn.ConvTranspose1d(**case)
        poly = kokoro_module._PolyphaseConvTranspose1d(ct)
        x = torch.randn(2, case["in_channels"], 37)
        with torch.no_grad():
            expected, actual = ct(x), poly(x)
        assert actual.shape == expected.shape, case
        assert torch.allclose(actual, expected, atol=1e-5), case


def test_fold_snake_alpha_is_exact() -> None:
    torch.manual_seed(0)
    block = kokoro_module.istftnet.AdaINResBlock1(16, 3, (1, 3, 5), style_dim=8)
    with torch.no_grad():
        for p in list(block.alpha1) + list(block.alpha2):
            p.uniform_(0.3, 2.0)
    kokoro_module._remove_weight_norm(block)
    x, s = torch.randn(1, 16, 50), torch.randn(1, 8)
    with torch.no_grad():
        expected = block(x, s)
        kokoro_module._fold_snake_alpha(block)
        actual = block(x, s)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_pipeline_is_built_once_per_language(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    built = []

    class DummyPipeline:
        def __init__(self, model, lang_code):
            built.append(lang_code)

    monkeypatch.setattr("kokoro.pipeline.KPipeline", DummyPipeline)
    kokoro = OV_Kokoro(load_config)

    first = kokoro._pipeline("a")
    assert kokoro._pipeline("a") is first
    kokoro._pipeline("b")

    assert built == ["a", "b"]


def test_exact_sinegen_matches_upstream_below_float_limit() -> None:
    # Same seed, same input: the size-based interpolation must reproduce the
    # upstream scale-factor version exactly where the latter is still exact.
    gen = kokoro_module.istftnet.SineGen(24000, upsample_scale=300, harmonic_num=8)
    f0 = torch.full((1, 300 * 40, 1), 180.0)
    f0_values = f0 * torch.arange(1, 10).view(1, 1, 9)

    torch.manual_seed(0)
    expected = kokoro_module._ORIGINAL_F02SINE(gen, f0_values.clone())
    torch.manual_seed(0)
    actual = kokoro_module._f02sine_exact(gen, f0_values.clone())

    assert torch.allclose(actual, expected, atol=1e-5)
    assert kokoro_module.istftnet.SineGen._f02sine is kokoro_module._ORIGINAL_F02SINE
    with kokoro_module._exact_sinegen():
        assert kokoro_module.istftnet.SineGen._f02sine is kokoro_module._f02sine_exact
    assert kokoro_module.istftnet.SineGen._f02sine is kokoro_module._ORIGINAL_F02SINE


def test_stream_chunks_shortens_only_the_first_chunk(load_config: ModelLoadConfig) -> None:
    kokoro = OV_Kokoro(load_config)
    first = "This opening sentence is deliberately long, so that it runs well past the streaming limit, which is about one hundred and twenty characters."
    text = first + " Second sentence. Third sentence here."

    chunks = kokoro._stream_chunks(text, 400)

    assert len(chunks[0]) <= kokoro_module.STREAM_FIRST_CHUNK_CHARS
    assert " ".join(chunks).split() == text.split()
    assert kokoro._stream_chunks("Short. Text.", 400) == kokoro.make_chunks("Short. Text.", 400)
