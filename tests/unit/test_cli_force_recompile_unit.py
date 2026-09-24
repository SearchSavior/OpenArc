"""Unit tests wiring the ``--force-recompile`` / ``--fr`` flag(s) of
``openarc serve start`` and ``openarc load`` to the server-side recompile path.

The flag is deliberately transported so it is NOT part of a model's persisted
configuration (which would reopen the config-hash gate on every subsequent
ordinary load):

* ``openarc load --force-recompile / --fr`` travels as a query parameter on the
  ``POST /openarc/load`` request (``?force_recompile=true``); the request body
  stays a plain load config.
* ``openarc serve start --force-recompile / --fr`` travels as the
  ``OPENARC_FORCE_RECOMPILE`` environment variable, which the server's lifespan
  reads and forwards to ``register_load(force_recompile=True)`` for every model
  loaded on startup.

These tests pin that contract: the flag sets exactly the expected wire value, and
its ABSENCE leaves the request/launch unchanged (byte-for-byte for the query,
``"false"`` for the env var) -- so a plain ``openarc load <model>`` /
``openarc serve start`` is exactly what it was before the flag existed.
"""

import json
import os

import pytest  # type: ignore[import]
from click.testing import CliRunner

import requests

import src.cli.modules.launch_server as launch_module
from src.cli import cli


class _Resp:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.text = ""


# --- helpers -----------------------------------------------------------------


def _seed_config(tmp_path, name="m1", device="CPU") -> str:
    """Create a temp openarc_config.json with one resolvable model entry.

    ``model_path`` is stored absolute so ``ServerConfig._resolve_model_paths``
    leaves it untouched, and a directory bearing ``openvino_model.{xml,bin}`` is
    created so ``validate_model_path`` accepts it. Returns the config file path
    (already pointed at via the conftest/test's env).
    """
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "openvino_model.xml").write_text(
        "<optimized_memory_network>", encoding="utf-8"
    )
    (model_dir / "openvino_model.bin").write_bytes(b"bin")

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
                        "device": device,
                        "runtime_config": {},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(cfg)


# --- openarc load --force-recompile / --fr -----------------------------------


def _run_load(cfg: str, monkeypatch: pytest.MonkeyPatch, *args: str):
    calls: list = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _Resp(200)

    monkeypatch.setattr(requests, "post", fake_post)
    # Set the config file (monkeypatch restores it) and also pass it through
    # click's env map so the command's ServerConfig() reads exactly this file.
    monkeypatch.setenv("OPENARC_CONFIG_FILE", cfg)
    result = CliRunner().invoke(cli, list(args), env={"OPENARC_CONFIG_FILE": cfg})
    return result, calls


def test_load_without_flag_sends_no_force_recompile_param(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain `openarc load <model>` must not send ``force_recompile`` at all --
    the query string is byte-for-byte unchanged from before the flag existed, so
    the request body and the persisted config are untouched."""
    cfg = _seed_config(tmp_path)
    result, calls = _run_load(cfg, monkeypatch, "load", "m1")

    assert result.exit_code == 0
    assert calls, "a load request must have been issued"
    # requests.post was called with params=None when the flag is absent.
    assert calls[-1].get("params") is None


def test_load_force_recompile_sends_param(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``openarc load <model> --force-recompile`` must send
    ``?force_recompile=true`` as a query parameter (not in the JSON body)."""
    cfg = _seed_config(tmp_path)
    result, calls = _run_load(cfg, monkeypatch, "load", "m1", "--force-recompile")

    assert result.exit_code == 0
    assert calls[-1].get("params") == {"force_recompile": "true"}
    # The load config the server hashes/persists must be the plain body, free of
    # the force flag.
    assert "force_recompile" not in calls[-1]["json"]


def test_load_fr_alias_sends_param(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--fr`` is an alias for ``--force-recompile`` on ``openarc load``."""
    cfg = _seed_config(tmp_path)
    result, calls = _run_load(cfg, monkeypatch, "load", "m1", "--fr")

    assert result.exit_code == 0
    assert calls[-1].get("params") == {"force_recompile": "true"}


def test_load_force_recompile_applies_to_every_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With multiple models on the line, the flag rides along on each of their
    per-model load requests (so every one of them is force-recompiled)."""
    cfg = _seed_config(tmp_path, name="a")
    # add a second model so a single --fr/--force-recompile must ride along on
    # BOTH per-model load requests
    model_dir_b = tmp_path / "models" / "b"
    model_dir_b.mkdir(parents=True, exist_ok=True)
    (model_dir_b / "openvino_model.xml").write_text("<x>", encoding="utf-8")
    (model_dir_b / "openvino_model.bin").write_bytes(b"bin")
    data = json.loads((tmp_path / "openarc_config.json").read_text(encoding="utf-8"))
    data["models"]["b"] = {
        "model_name": "b",
        "model_path": str(model_dir_b),
        "model_type": "llm",
        "engine": "ovgenai",
        "device": "CPU",
        "runtime_config": {},
    }
    (tmp_path / "openarc_config.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )

    calls: list = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _Resp(200)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OPENARC_CONFIG_FILE", cfg)
    result = CliRunner().invoke(
        cli, ["load", "a", "b", "--fr"], env={"OPENARC_CONFIG_FILE": cfg}
    )

    assert result.exit_code == 0
    assert len(calls) == 2
    assert all(c.get("params") == {"force_recompile": "true"} for c in calls)


# --- openarc serve start --force-recompile / --fr ----------------------------


def _run_serve(cfg: str, monkeypatch: pytest.MonkeyPatch, *args: str):
    captured: list = []

    def fake_start(host=None, port=None, verbose=0, reload=False):
        # The command sets OPENARC_FORCE_RECOMPILE in os.environ just before this
        # call, and click only reverts env keys it was given explicitly -- so this
        # read reflects exactly what the command decided to set.
        captured.append(
            {
                "host": host,
                "port": port,
                "force_recompile": os.environ.get("OPENARC_FORCE_RECOMPILE"),
                "startup_models": os.environ.get("OPENARC_STARTUP_MODELS"),
            }
        )
        return None

    monkeypatch.setattr(launch_module, "start_server", fake_start)
    monkeypatch.setenv("OPENARC_CONFIG_FILE", cfg)
    result = CliRunner().invoke(cli, list(args), env={"OPENARC_CONFIG_FILE": cfg})
    return result, captured


def test_serve_start_defaults_force_recompile_env_to_false(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the flag, ``serve start`` tells the server NOT to force a
    recompile (OPENARC_FORCE_RECOMPILE="false"), so startup loads keep their warm
    cache when the config is unchanged."""
    cfg = _seed_config(tmp_path, name="sm")
    result, captured = _run_serve(cfg, monkeypatch, "serve", "start")

    assert result.exit_code == 0
    assert captured, "start_server must have been invoked"
    assert captured[-1]["force_recompile"] == "false"


def test_serve_start_sets_force_recompile_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``openarc serve start --force-recompile --load-models <models>`` sets
    OPENARC_FORCE_RECOMPILE="true", which the server forwards to register_load for
    every model loaded on startup."""
    cfg = _seed_config(tmp_path, name="sm")
    result, captured = _run_serve(
        cfg, monkeypatch, "serve", "start", "--force-recompile", "--load-models", "sm"
    )

    assert result.exit_code == 0
    assert captured[-1]["force_recompile"] == "true"
    assert captured[-1]["startup_models"] == "sm"


def test_serve_start_fr_alias_sets_force_recompile_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--fr`` is an alias for ``--force-recompile`` on ``openarc serve start``."""
    cfg = _seed_config(tmp_path, name="sm")
    result, captured = _run_serve(
        cfg, monkeypatch, "serve", "start", "--fr", "--load-models", "sm"
    )

    assert result.exit_code == 0
    assert captured[-1]["force_recompile"] == "true"
    assert captured[-1]["startup_models"] == "sm"
