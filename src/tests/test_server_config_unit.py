from pathlib import Path

from src.cli.modules.server_config import ServerConfig


def _config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(config_file=tmp_path / "openarc_config.json")


def test_resolve_cache_dir_relative_to_config_dir(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    resolved = cfg._resolve_model_paths({"model_path": "/abs/model", "cache_dir": "model_cache"})
    assert resolved["cache_dir"] == str((tmp_path / "model_cache").resolve())


def test_resolve_cache_dir_absolute_passthrough(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    resolved = cfg._resolve_model_paths({"cache_dir": "/tmp/ov_cache"})
    assert resolved["cache_dir"] == "/tmp/ov_cache"


def test_resolve_cache_dir_absent(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    resolved = cfg._resolve_model_paths({"model_path": "/abs/model"})
    assert "cache_dir" not in resolved
