import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch
import pytest  # type: ignore[import]

import src.engine.optimum.optimum_emb as emb_module
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType
from src.server.schemas.modeling.contract_optimum_emb import PreTrainedTokenizerConfig


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
        ov_config=load_config.runtime_config,
    )
    emb_module.AutoTokenizer.from_pretrained.assert_called_once_with(load_config.model_path)
    assert emb.model is model_instance
    assert emb.tokenizer is tokenizer_instance


def test_load_model_forwards_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = ModelLoadConfig(
        model_path="/models/mock",
        model_name="cache-model",
        model_type=ModelType.EMB,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
        cache_dir="/tmp/ov_cache",
    )
    emb = Optimum_EMB(loader)
    monkeypatch.setattr(
        emb_module.OVModelForFeatureExtraction,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(emb_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=MagicMock()))

    emb.load_model(loader)

    emb_module.OVModelForFeatureExtraction.from_pretrained.assert_called_once_with(
        loader.model_path,
        device=loader.device,
        export=False,
        ov_config={"CACHE_DIR": "/tmp/ov_cache"},
    )


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



def test_position_ids_from_mask_right_and_left_padding() -> None:
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])

    assert Optimum_EMB.position_ids_from_mask(right).tolist() == [[0, 1, 2, 1, 1], [0, 1, 2, 3, 4]]
    assert Optimum_EMB.position_ids_from_mask(left).tolist() == [[1, 1, 0, 1, 2], [0, 1, 2, 3, 4]]


class _Batch(dict):
    def to(self, device):  # mimic HuggingFace BatchEncoding
        return self


def _tok_config() -> PreTrainedTokenizerConfig:
    return PreTrainedTokenizerConfig(text=["a", "bbb"], padding=True, truncation=True, max_length=16)


def _run_embeddings(emb: Optimum_EMB, tok_config: PreTrainedTokenizerConfig):
    async def _run():
        return [item async for item in emb.generate_embeddings(tok_config)]

    return asyncio.run(_run())


def test_generate_embeddings_feeds_position_ids_when_ir_expects_them(load_config: ModelLoadConfig) -> None:
    """Decoder IRs (pre optimum-intel 1.26 Qwen3-Embedding exports) take position_ids."""
    emb = Optimum_EMB(load_config)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    emb.tokenizer = MagicMock(
        return_value=_Batch({"input_ids": torch.tensor([[5, 6, 0], [7, 8, 9]]), "attention_mask": attention_mask})
    )
    hidden = torch.tensor([
        [[1.0, 0.0], [3.0, 4.0], [9.0, 9.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
    ])
    model = MagicMock()
    model.device = "cpu"
    model.input_names = {"input_ids": 0, "attention_mask": 1, "position_ids": 2}
    model._inference.return_value = {"last_hidden_state": hidden.numpy()}
    emb.model = model

    vectors = _run_embeddings(emb, _tok_config())

    model.compile.assert_called_once()
    model.assert_not_called()  # forward() would drop position_ids
    sent = model._inference.call_args.args[0]
    assert set(sent) == {"input_ids", "attention_mask", "position_ids"}
    assert sent["position_ids"].tolist() == [[0, 1, 1], [0, 1, 2]]
    # last real token of each row, normalized
    assert torch.allclose(torch.tensor(vectors[0]), torch.tensor([[0.6, 0.8], [0.0, 1.0]]))


def test_generate_embeddings_uses_forward_for_encoder_irs(load_config: ModelLoadConfig) -> None:
    """IRs without position_ids (BERT-style, current optimum-intel exports) keep the forward() path."""
    emb = Optimum_EMB(load_config)
    attention_mask = torch.tensor([[1, 1]])
    emb.tokenizer = MagicMock(
        return_value=_Batch({"input_ids": torch.tensor([[1, 2]]), "attention_mask": attention_mask})
    )
    model = MagicMock()
    model.device = "cpu"
    model.input_names = {"input_ids": 0, "attention_mask": 1}
    model.return_value = SimpleNamespace(last_hidden_state=torch.tensor([[[1.0, 0.0], [0.0, 3.0]]]))
    emb.model = model

    vectors = _run_embeddings(emb, _tok_config())

    model.assert_called_once()
    model._inference.assert_not_called()
    assert vectors[0] == [[0.0, 1.0]]


def test_position_ids_path_also_fills_token_type_ids(load_config: ModelLoadConfig) -> None:
    emb = Optimum_EMB(load_config)
    emb.tokenizer = MagicMock(
        return_value=_Batch({"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.tensor([[1, 1]])})
    )
    model = MagicMock()
    model.device = "cpu"
    model.input_names = {"input_ids": 0, "attention_mask": 1, "position_ids": 2, "token_type_ids": 3}
    model._inference.return_value = {"last_hidden_state": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).numpy()}
    emb.model = model

    _run_embeddings(emb, _tok_config())

    sent = model._inference.call_args.args[0]
    assert sent["token_type_ids"].tolist() == [[0, 0]]


@pytest.mark.parametrize(
    "input_names, expect_info, expect_warning",
    [
        ({"input_ids": 0, "attention_mask": 1}, False, False),
        ({"input_ids": 0, "attention_mask": 1, "position_ids": 2}, True, False),
        ({"input_ids": 0, "attention_mask": 1, "beam_idx": 2}, False, True),
    ],
)
def test_load_logs_how_inputs_are_handled(
    load_config: ModelLoadConfig, caplog: pytest.LogCaptureFixture, input_names, expect_info, expect_warning
) -> None:
    emb = Optimum_EMB(load_config)
    emb.model = SimpleNamespace(input_names=input_names)

    with caplog.at_level("INFO"):
        emb._log_input_handling("test-emb")

    assert ("takes position_ids" in caplog.text) is expect_info
    assert ("does not fill" in caplog.text) is expect_warning
