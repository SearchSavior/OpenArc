"""Unit tests for the server-side max_tokens default.

Covers:
* ``_apply_model_max_tokens_default`` -- the pure helper that substitutes the
  model-level max_tokens for a request that omits it (and never overrides an
  explicit client value).
* ``ModelRegistry.register_load`` -- that the model-level max_tokens is stored on
  the record so the routes can read it.
* ``_model_max_tokens`` -- the per-model lookup used by the routes.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
import src.server.routes.openai as openai_module
from src.server.model_registry import ModelRegistry
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType


def _load_config(name: str, model_path: str, max_tokens: int | None = None) -> ModelLoadConfig:
    kwargs = dict(
        model_path=model_path,
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
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


# --- the pure helper --------------------------------------------------------
def test_apply_model_max_tokens_uses_model_default_when_client_omits() -> None:
    gen = OVGenAI_GenConfig(messages=[{"role": "user", "content": "hi"}])
    assert gen.max_tokens == 16384  # the large default we are guarding against
    out = openai_module._apply_model_max_tokens_default(gen, None, 1024)
    assert out.max_tokens == 1024


def test_apply_model_max_tokens_keeps_explicit_client_value() -> None:
    gen = OVGenAI_GenConfig(messages=[{"role": "user", "content": "hi"}], max_tokens=512)
    out = openai_module._apply_model_max_tokens_default(gen, 512, 1024)
    assert out.max_tokens == 512


def test_apply_model_max_tokens_keeps_default_when_model_unset() -> None:
    gen = OVGenAI_GenConfig(messages=[{"role": "user", "content": "hi"}])
    out = openai_module._apply_model_max_tokens_default(gen, None, None)
    assert out.max_tokens == 16384


# --- record propagation -----------------------------------------------------
def test_register_load_stores_max_tokens_on_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("m", str(tmp_path), max_tokens=2048))
    record = next(r for r in registry._models.values() if r.model_name == "m")
    assert record.max_tokens == 2048


def test_register_load_max_tokens_defaults_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("m", str(tmp_path)))
    record = next(r for r in registry._models.values() if r.model_name == "m")
    assert record.max_tokens is None


def test_model_max_tokens_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = ModelRegistry()
    _register(monkeypatch, registry, _load_config("m", str(tmp_path), max_tokens=777))
    monkeypatch.setattr(openai_module, "_registry", registry)

    async def _run():
        found = await openai_module._model_max_tokens("m")
        missing = await openai_module._model_max_tokens("missing")
        return found, missing

    found, missing = asyncio.run(_run())
    assert found == 777
    assert missing is None
