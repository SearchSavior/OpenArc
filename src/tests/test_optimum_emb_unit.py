import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch
import pytest  # type: ignore[import]

import src.engine.optimum.optimum_emb as emb_module
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.model_registry import EngineType, ModelLoadConfig, ModelType
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


def test_generate_embeddings_returns_normalized_vectors(load_config: ModelLoadConfig) -> None:
    emb = Optimum_EMB(load_config)

    class DummyBatch(dict):
        def to(self, device):  # noqa: D401 - mimic HuggingFace BatchEncoding
            self["moved_to"] = device
            return self

    attention_mask = torch.tensor([[1, 1, 1]])
    batch = DummyBatch({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": attention_mask})

    tokenizer_mock = MagicMock(return_value=batch)
    emb.tokenizer = tokenizer_mock

    outputs = torch.tensor([
        [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]],
    ])

    class DummyModel:
        device = "cpu"

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=outputs)

    emb.model = DummyModel()

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
    tokenizer_mock.assert_called_once()


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

