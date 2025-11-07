import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch
import pytest  # type: ignore[import]

import src.engine.optimum.optimum_rr as rr_module
from src.engine.optimum.optimum_rr import Optimum_RR
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.optimum import RerankerConfig


MODEL_PATH = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-Reranker-0.6B-fp16-ov"


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=MODEL_PATH,
        model_name="test-rerank",
        model_type=ModelType.RERANK,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={"config": "value"},
    )


def test_compute_logits_returns_probabilities(load_config: ModelLoadConfig) -> None:
    rr = Optimum_RR(load_config)
    rr.token_true_id = 1
    rr.token_false_id = 0

    logits = torch.tensor([
        [[0.0, 0.0, 0.0], [0.2, 0.8, -0.5]],
    ])

    class DummyModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)

    rr.model = DummyModel()

    inputs = {"input_ids": torch.ones((1, 2), dtype=torch.long)}
    scores = rr.compute_logits(inputs)

    assert isinstance(scores, list)
    assert len(scores) == 1
    assert scores[0] > 0


def test_generate_rerankings_orders_documents(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    rr = Optimum_RR(load_config)
    rr.token_true_id = 1
    rr.token_false_id = 0

    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=False):
            return [101]

        def __call__(self, pairs, **kwargs):
            self.calls.append(("call", pairs, kwargs))
            return {"input_ids": [[10, 11], [20, 21]]}

        def pad(self, inputs, padding=True, return_tensors="pt", max_length=None):
            return {
                "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
                "attention_mask": torch.ones(len(inputs["input_ids"]), len(inputs["input_ids"][0])),
            }

        def convert_tokens_to_ids(self, token):
            return 1 if token == "yes" else 0

    rr.tokenizer = DummyTokenizer()

    logits = torch.tensor([
        [[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]],
        [[0.0, 0.0, 0.0], [0.5, 1.5, -0.2]],
    ])

    class DummyModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)

    rr.model = DummyModel()

    config = RerankerConfig(
        query="capital of france",
        documents=["Paris is capital of France.", "Berlin is capital of Germany."],
        prefix="",
        suffix="",
        instruction="Pick matching document",
        max_length=32,
    )

    async def _run():
        results = []
        async for item in rr.generate_rerankings(config):
            results.append(item)
        return results

    ranked = asyncio.run(_run())[0]

    assert ranked[0]["doc"].startswith("Paris")
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_load_model_initializes_pipeline(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    rr = Optimum_RR(load_config)
    model_instance = MagicMock()
    monkeypatch.setattr(
        rr_module.OVModelForCausalLM,
        "from_pretrained",
        MagicMock(return_value=model_instance),
    )

    tokenizer_instance = MagicMock()
    tokenizer_instance.convert_tokens_to_ids.side_effect = lambda token: {"no": 0, "yes": 1}[token]
    monkeypatch.setattr(rr_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer_instance))

    rr.load_model(load_config)

    rr_module.OVModelForCausalLM.from_pretrained.assert_called_once_with(
        load_config.model_path,
        device=load_config.device,
        export=False,
        use_cache=False,
    )
    rr_module.AutoTokenizer.from_pretrained.assert_called_once_with(load_config.model_path)
    assert rr.model is model_instance
    assert rr.tokenizer is tokenizer_instance
    assert rr.token_true_id == 1
    assert rr.token_false_id == 0


def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    rr = Optimum_RR(load_config)
    rr.model = object()
    rr.tokenizer = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(rr_module.gc, "collect", gc_mock)

    result = asyncio.run(rr.unload_model(registry, "model-name"))

    assert result is True
    assert rr.model is None
    assert rr.tokenizer is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()

