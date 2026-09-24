"""Unit tests for the /v1/models context-window propagation.

These exercise the actual ``openai_list_models`` handler end to end: a model is
registered in a real ``ModelRegistry`` (with a fake factory, so no heavy engine
is loaded), then the OpenAI-compatible ``/v1/models`` response is built. The
assertions confirm the two fields that OpenArc propagates to chat clients:

* ``context_window`` -- the OpenAI-standard field clients read to size
  context / auto-compaction, and
* ``meta.n_ctx`` -- the non-standard field goose reads from /v1/models.

When neither can be resolved (no config.json, no explicit value) the response
must omit both so clients fall back to their own defaults.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
import src.server.routes.openai as openai_module
from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelType,
)


def _write_model_dir(tmp_path: Path, name: str, payload: dict | None) -> str:
    """Create a fake model directory; only writes config.json when ``payload`` is set."""
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True)
    if payload is not None:
        (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(model_dir)


def _load_config(name: str, model_path: str, context_window: int | None = None) -> ModelLoadConfig:
    kwargs = dict(
        model_path=model_path,
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )
    if context_window is not None:
        kwargs["context_window"] = context_window
    return ModelLoadConfig(**kwargs)


def _register(monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry, load_config: ModelLoadConfig):
    async def _noop_unload(*_args, **_kwargs):
        return None

    async def fake_create(config):  # type: ignore[override]
        return SimpleNamespace(unload_model=_noop_unload)

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)

    asyncio.run(_run())


def test_list_models_emits_context_window_and_meta_ncx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # max_position_embeddings precedes n_ctx in the priority list, so it wins.
    model_path = _write_model_dir(
        tmp_path, "ctx-model", {"max_position_embeddings": 131072, "n_ctx": 32768}
    )
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("ctx-model", model_path))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())

    assert response["object"] == "list"
    entry = response["data"][0]
    assert entry["id"] == "ctx-model"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "OpenArc"
    # Both the standard field and goose's meta.n_ctx are emitted from one value.
    assert entry["context_window"] == 131072
    assert entry["meta"] == {"n_ctx": 131072}


def test_list_models_omits_context_fields_when_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No config.json, no explicit value -> context_window is None -> omit both.
    model_path = _write_model_dir(tmp_path, "no-ctx", None)
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("no-ctx", model_path))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())
    entry = response["data"][0]
    assert entry["id"] == "no-ctx"
    assert "context_window" not in entry
    assert "meta" not in entry


def test_list_models_explicit_override_flows_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An explicit value must be what the response advertises (and what meta.n_ctx mirrors).
    model_path = _write_model_dir(tmp_path, "override", {"max_position_embeddings": 9999})
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("override", model_path, context_window=5000))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())
    entry = response["data"][0]
    assert entry["context_window"] == 5000
    assert entry["meta"] == {"n_ctx": 5000}


def test_list_models_mixed_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    known = _write_model_dir(tmp_path, "known", {"seq_len": 6000})
    unknown = _write_model_dir(tmp_path, "unknown", None)
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("known", known))
    _register(monkeypatch, registry, _load_config("unknown", unknown))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())
    entries = {e["id"]: e for e in response["data"]}

    assert entries["known"]["context_window"] == 6000
    assert entries["known"]["meta"] == {"n_ctx": 6000}
    assert "context_window" not in entries["unknown"]
    assert "meta" not in entries["unknown"]


def test_list_models_discovers_nested_text_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The reported symptom, end to end: a multimodal / VLM config.json nests the
    # window under ``text_config`` (and carries a decoy top-level sliding_window).
    # A flat top-level scan would advertise nothing; the section-aware scan must
    # surface the real window in BOTH fields clients read.
    model_path = _write_model_dir(
        tmp_path,
        "qwen25vl",
        {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "sliding_window": 512,  # top-level decoy (lower priority)
            "text_config": {
                "model_type": "qwen2_5_vl_text",
                "max_position_embeddings": 32768,
            },
            "vision_config": {"model_type": "qwen2_5_vl"},
        },
    )
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("qwen25vl", model_path))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        return await openai_module.openai_list_models()

    response = asyncio.run(_run())
    entry = response["data"][0]
    assert entry["id"] == "qwen25vl"
    # Both advertised fields surface the nested (text_config) value, not the decoy.
    assert entry["context_window"] == 32768
    assert entry["meta"] == {"n_ctx": 32768}
