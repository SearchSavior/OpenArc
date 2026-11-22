import asyncio
from unittest.mock import AsyncMock

import pytest  # type: ignore[import]
from fastapi import HTTPException

import src.server.routes.openai as openai_module
from src.server.schemas.modeling.contract_optimum_emb import (
    EMB_TOKENIZER_DEFAULTS,
    PreTrainedTokenizerConfig,
    shorten_embedding,
)
from src.server.schemas.requests_openai import EmbeddingsRequest


def _call(monkeypatch: pytest.MonkeyPatch, data, **request_fields):
    embed_mock = AsyncMock(return_value={"data": data, "metrics": {}})
    monkeypatch.setattr(openai_module, "_workers", type("W", (), {"embed": embed_mock})())
    request = EmbeddingsRequest(model="emb", **request_fields)
    response = asyncio.run(openai_module.embeddings(request))
    return response, embed_mock.call_args.args[1]


def test_plain_openai_batch_gets_padding_and_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAI client sends only model + input; the batch must still be paddable."""
    response, tok_config = _call(monkeypatch, [[1.0], [1.0]], input=["short", "a much longer text"])

    assert tok_config.text == ["short", "a much longer text"]
    assert tok_config.padding is True
    assert tok_config.truncation is True
    assert tok_config.max_length == EMB_TOKENIZER_DEFAULTS["max_length"]
    assert [item["index"] for item in response["data"]] == [0, 1]


def test_client_config_values_win_over_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PreTrainedTokenizerConfig(padding="max_length", max_length=64)
    _, tok_config = _call(monkeypatch, [[1.0]], input="hi", config=config)

    assert tok_config.padding == "max_length"
    assert tok_config.max_length == 64
    assert tok_config.truncation is True  # unset by the client, so the default applies
    assert tok_config.text == "hi"


def test_dimensions_shortens_output_not_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI `dimensions` is the output vector size, not a tokenizer max_length."""
    response, tok_config = _call(monkeypatch, [[3.0, 4.0, 12.0]], input="hi", dimensions=2)

    assert tok_config.max_length == EMB_TOKENIZER_DEFAULTS["max_length"]
    assert response["data"][0]["embedding"] == pytest.approx([0.6, 0.8])


def test_dimensions_larger_than_model_is_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, [[1.0, 0.0]], input="hi", dimensions=3)
    assert exc.value.status_code == 400


def test_non_positive_dimensions_is_a_400_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_mock = AsyncMock()
    monkeypatch.setattr(openai_module, "_workers", type("W", (), {"embed": embed_mock})())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(openai_module.embeddings(EmbeddingsRequest(model="emb", input="hi", dimensions=0)))
    assert exc.value.status_code == 400
    embed_mock.assert_not_called()


def test_shorten_embedding_renormalizes() -> None:
    assert shorten_embedding([3.0, 4.0, 5.0], 2) == pytest.approx([0.6, 0.8])
    assert shorten_embedding([0.0, 0.0, 1.0], 2) == [0.0, 0.0]
