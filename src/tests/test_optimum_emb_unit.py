import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch
import pytest  # type: ignore[import]

import src.engine.optimum.optimum_emb as emb_module
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import PreTrainedTokenizerConfig


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/Pytorch/Qwen/Qwen3-Embed-0.6B-INT8-ASYM-ov"


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-emb",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={"config": "value"},
    )


def test_last_token_pool_uses_sequence_length(load_config: ModelLoadConfig) -> None:
    emb = Optimum_EMB(load_config)
    states = torch.tensor([
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
    ])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]])

    pooled = emb.last_token_pool(states, attention)

    expected = torch.tensor([[4.0, 5.0], [8.0, 9.0]])
    assert torch.equal(pooled, expected)


def test_cls_pool_returns_first_token() -> None:
    states = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ])
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = Optimum_EMB.cls_pool(states, attention)

    expected = torch.tensor([[1.0, 2.0], [7.0, 8.0]])
    assert torch.equal(pooled, expected)


def test_mean_pool_ignores_padding() -> None:
    states = torch.tensor([
        [[2.0, 4.0], [4.0, 8.0], [100.0, 200.0]],  # last token is padding
    ])
    attention = torch.tensor([[1, 1, 0]])

    pooled = Optimum_EMB.mean_pool(states, attention)

    expected = torch.tensor([[3.0, 6.0]])
    assert torch.allclose(pooled, expected)


def test_detect_pool_mode_defaults_to_last(tmp_path: Path) -> None:
    # No 1_Pooling/ dir => last-token (Qwen3-Embedding behavior preserved)
    assert Optimum_EMB._detect_pool_mode(str(tmp_path)) == "last"


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({"pooling_mode_cls_token": True}, "cls"),
        ({"pooling_mode_mean_tokens": True}, "mean"),
        ({"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": False}, "last"),
    ],
)
def test_detect_pool_mode_reads_sentence_transformers_config(
    tmp_path: Path, cfg: dict, expected: str
) -> None:
    pool_dir = tmp_path / "1_Pooling"
    pool_dir.mkdir()
    (pool_dir / "config.json").write_text(json.dumps(cfg))

    assert Optimum_EMB._detect_pool_mode(str(tmp_path)) == expected


def _make_emb_with_dummy_pipeline(
    load_config: ModelLoadConfig,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_mode: str = "last",
):
    emb = Optimum_EMB(load_config)
    emb.pool_mode = pool_mode

    class DummyBatch(dict):
        def to(self, device):  # noqa: D401 - mimic HuggingFace BatchEncoding
            self["moved_to"] = device
            return self

    batch = DummyBatch({"input_ids": torch.zeros_like(attention_mask), "attention_mask": attention_mask})
    emb.tokenizer = MagicMock(return_value=batch)

    class DummyModel:
        device = "cpu"

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=hidden_states)

    emb.model = DummyModel()
    return emb


def test_generate_embeddings_returns_normalized_vectors(load_config: ModelLoadConfig) -> None:
    outputs = torch.tensor([
        [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]],
    ])
    attention_mask = torch.tensor([[1, 1, 1]])
    emb = _make_emb_with_dummy_pipeline(load_config, outputs, attention_mask, pool_mode="last")

    tok_config = PreTrainedTokenizerConfig(
        text=["hello world"],
        padding="longest",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )

    async def _run():
        collected = []
        async for item in emb.generate_embeddings(tok_config):
            collected.append(item)
        return collected

    vectors = asyncio.run(_run())

    assert len(vectors) == 1
    vec = torch.tensor(vectors[0])
    assert torch.allclose(torch.norm(vec, dim=1), torch.ones(1), atol=1e-6)


def test_generate_embeddings_uses_cls_pool_when_configured(load_config: ModelLoadConfig) -> None:
    # First token is the CLS; make it obviously distinct from the others.
    outputs = torch.tensor([
        [[3.0, 4.0], [100.0, 100.0], [-100.0, -100.0]],
    ])
    attention_mask = torch.tensor([[1, 1, 1]])
    emb = _make_emb_with_dummy_pipeline(load_config, outputs, attention_mask, pool_mode="cls")

    tok_config = PreTrainedTokenizerConfig(
        text=["hello world"],
        padding="longest",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )

    async def _run():
        collected = []
        async for item in emb.generate_embeddings(tok_config):
            collected.append(item)
        return collected

    vectors = asyncio.run(_run())
    # [3, 4] normalized = [0.6, 0.8]
    assert torch.allclose(torch.tensor(vectors[0]), torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_load_model_initializes_pipeline(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    emb = Optimum_EMB(load_config)
    model_instance = MagicMock()
    monkeypatch.setattr(
        emb_module.OVModelForFeatureExtraction,
        "from_pretrained",
        MagicMock(return_value=model_instance),
    )

    tokenizer_instance = MagicMock()
    monkeypatch.setattr(emb_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer_instance))

    emb.load_model(load_config)

    emb_module.OVModelForFeatureExtraction.from_pretrained.assert_called_once_with(
        load_config.model_path,
        device=load_config.device,
        export=False,
    )
    emb_module.AutoTokenizer.from_pretrained.assert_called_once_with(load_config.model_path)
    assert emb.model is model_instance
    assert emb.tokenizer is tokenizer_instance
    # Nonexistent path => default pooling preserved.
    assert emb.pool_mode == "last"


def test_load_model_picks_up_cls_pool_from_sentence_transformers_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool_dir = tmp_path / "1_Pooling"
    pool_dir.mkdir()
    (pool_dir / "config.json").write_text(json.dumps({"pooling_mode_cls_token": True}))

    cfg = ModelLoadConfig(
        model_path=str(tmp_path),
        model_name="bge-m3-like",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )
    emb = Optimum_EMB(cfg)
    monkeypatch.setattr(
        emb_module.OVModelForFeatureExtraction, "from_pretrained", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(emb_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=MagicMock()))

    emb.load_model(cfg)

    assert emb.pool_mode == "cls"


def test_runtime_config_pool_mode_override_beats_autodetect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Model ships sentence-transformers metadata saying CLS...
    pool_dir = tmp_path / "1_Pooling"
    pool_dir.mkdir()
    (pool_dir / "config.json").write_text(json.dumps({"pooling_mode_cls_token": True}))

    # ...but the operator pins last-token pooling via runtime_config.
    cfg = ModelLoadConfig(
        model_path=str(tmp_path),
        model_name="override-wins",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={"pool_mode": "last"},
    )
    emb = Optimum_EMB(cfg)
    monkeypatch.setattr(
        emb_module.OVModelForFeatureExtraction, "from_pretrained", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(emb_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=MagicMock()))

    emb.load_model(cfg)

    assert emb.pool_mode == "last"


def test_load_model_rejects_unknown_pool_mode_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ModelLoadConfig(
        model_path=str(tmp_path),
        model_name="typo",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={"pool_mode": "clas"},  # typo of "cls"
    )
    emb = Optimum_EMB(cfg)
    monkeypatch.setattr(
        emb_module.OVModelForFeatureExtraction, "from_pretrained", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(emb_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=MagicMock()))

    with pytest.raises(ValueError, match="pool_mode"):
        emb.load_model(cfg)


def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    emb = Optimum_EMB(load_config)
    emb.model = object()
    emb.tokenizer = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(emb_module.gc, "collect", gc_mock)

    result = asyncio.run(emb.unload_model(registry, "model-name"))

    assert result is True
    assert emb.model is None
    assert emb.tokenizer is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

