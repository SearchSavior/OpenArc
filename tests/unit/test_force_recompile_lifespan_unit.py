"""Unit tests for the server-side half of ``--force-recompile`` / ``--fr``.

``openarc serve start --force-recompile`` sets ``OPENARC_FORCE_RECOMPILE``; the
server's startup :func:`~src.server.main.lifespan` reads it and, for every model
loaded on startup, forwards ``force_recompile`` to
:func:`~src.server.model_registry.ModelRegistry.register_load`. That is the exact
same entry point -- and the exact same ``force_recompile`` parameter -- that
``openarc load --fr`` reaches via ``POST /openarc/load?force_recompile=true``,
so the recompile the operator gets is identical whether requested at startup or
on a running server. These tests pin that bridge (env var -> register_load call).
"""

import json
from pathlib import Path

import pytest  # type: ignore[import]

import src.server.main as main


class _RecordingRegistry:
    """Stand-in for the real ``ModelRegistry`` that records what the lifespan
    hands to ``register_load`` -- without touching the engine or the config-hash
    gate, so the test isolates the env-var -> ``force_recompile`` forwarding.
    """

    def __init__(self):
        self.calls: list = []

    async def register_load(self, loader, force_recompile=False):
        self.calls.append((loader.model_name, force_recompile))
        return loader.model_name


class _Application:
    """Unused placeholder for the lifespan ``app`` parameter (the body never
    references it); avoids standing up the real FastAPI app for the test."""


def _startup_config(tmp_path: Path, name: str = "m1") -> Path:
    """Temp openarc_config.json with one absolute-path model entry the lifespan
    resolves and loads on startup."""
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True)
    cfg = tmp_path / "openarc_config.json"
    cfg.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 8123},
                "models": {
                    name: {
                        "model_name": name,
                        "model_path": str(model_dir),
                        "model_type": "llm",
                        "engine": "ovgenai",
                        "device": "CPU",
                        "runtime_config": {},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cfg


def _drive_lifespan(monkeypatch: pytest.MonkeyPatch, cfg: Path, enabled: bool | None):
    monkeypatch.setenv("OPENARC_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("OPENARC_STARTUP_MODELS", "m1")
    if enabled is None:
        monkeypatch.delenv("OPENARC_FORCE_RECOMPILE", raising=False)
    else:
        monkeypatch.setenv("OPENARC_FORCE_RECOMPILE", "true" if enabled else "false")

    registry = _RecordingRegistry()
    monkeypatch.setattr(main, "_registry", registry)

    import asyncio

    async def _run():
        async with main.lifespan(_Application()):
            pass

    asyncio.run(_run())
    return registry


def test_startup_forwards_force_recompile_when_env_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OPENARC_FORCE_RECOMPILE=true`` -> the startup load is registered with
    ``force_recompile=True`` (an explicit operator request, not a config change)."""
    registry = _drive_lifespan(monkeypatch, _startup_config(tmp_path), enabled=True)
    assert registry.calls == [("m1", True)]


def test_startup_does_not_force_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag unset (the default), a startup load must NOT be force-
    recompiled: ``force_recompile`` defaults to False and the server instead
    relies on the config-hash gate (warm cache reused when the config is
    unchanged)."""
    registry = _drive_lifespan(monkeypatch, _startup_config(tmp_path), enabled=None)
    assert registry.calls == [("m1", False)]


def test_no_startup_models_registers_nothing_even_if_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag alone (no startup models) must not register or force anything:
    ``force_recompile`` only governs models that are actually loaded on startup.
    Pinned so ``openarc serve start --fr`` with no ``--load-models`` is a no-op
    with respect to loading, not a spurious forced recompile of nothing."""
    monkeypatch.setenv("OPENARC_CONFIG_FILE", str(_startup_config(tmp_path)))
    monkeypatch.setenv("OPENARC_STARTUP_MODELS", "")
    monkeypatch.setenv("OPENARC_FORCE_RECOMPILE", "true")

    registry = _RecordingRegistry()
    monkeypatch.setattr(main, "_registry", registry)

    import asyncio

    async def _run():
        async with main.lifespan(_Application()):
            pass

    asyncio.run(_run())

    assert registry.calls == []
