import yaml

from click.testing import CliRunner

from src.cli import cli


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml />", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_bytes(b"bin")
    return model_dir


def test_add_help_omits_vlm_type_option() -> None:
    result = CliRunner().invoke(cli, ["add", "--help"])

    assert result.exit_code == 0
    assert "--vlm-type" not in result.output
    assert "--vt" not in result.output


def _invoke_add(config_file, model_dir, *extra):
    from click.testing import CliRunner

    return CliRunner().invoke(
        cli,
        [
            "add",
            "--model-name",
            "limit-model",
            "--model-path",
            str(model_dir),
            "--engine",
            "ovgenai",
            "--model-type",
            "llm",
            "--device",
            "CPU",
            *extra,
        ],
        env={"OPENARC_CONFIG_FILE": str(config_file)},
    )


def test_add_worker_line_limit_with_suffix(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(
        config_file, model_dir, "--worker-line-limit", "512K"
    )
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert config["models"]["limit-model"]["load_config"]["worker_line_limit"] == 512 * 1024


def test_add_worker_line_limit_suffix_variants_and_plain_bytes(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    for value, expected in [
        ("256MiB", 256 * 1024**2),
        ("1g", 1024**3),
        ("131072", 131072),
    ]:
        result = _invoke_add(config_file, model_dir, "--worker-line-limit", value)
        assert result.exit_code == 0, result.output
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert config["models"]["limit-model"]["load_config"]["worker_line_limit"] == expected


def test_add_worker_line_limit_rejects_invalid_and_too_small(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    for value in ["abc", "12X", "1K"]:  # 1K = 1024 < 64 KiB floor
        result = _invoke_add(config_file, model_dir, "--worker-line-limit", value)
        assert result.exit_code == 1
        assert "worker-line-limit" in result.output
    assert not config_file.exists()


def test_add_omits_worker_line_limit_when_unspecified(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(config_file, model_dir)
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert "worker_line_limit" not in config["models"]["limit-model"]["load_config"]


def test_add_worker_max_respawns_positive_is_saved(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(config_file, model_dir, "--worker-max-respawns", "5")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert (
        config["models"]["limit-model"]["load_config"]["worker_max_respawns"]
        == 5
    )


def test_add_worker_max_respawns_zero_is_saved(tmp_path) -> None:
    """``0`` is the "no limit" sentinel, not the absence of the setting: click
    coerces it to ``0`` and it must round-trip as ``0`` (not be dropped as if
    unset, not error as a rejected lower bound)."""
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(config_file, model_dir, "--worker-max-respawns", "0")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    # 0 must round-trip as the integer 0, not be coalesced to None (absent).
    assert (
        config["models"]["limit-model"]["load_config"]["worker_max_respawns"] == 0
    )


def test_add_worker_max_respawns_negative_is_saved(tmp_path) -> None:
    """A negative number is also "no limit" (the worker is always reloaded and
    never quarantined). It must be accepted as a value -- click coerces
    ``--worker-max-respawns -1`` to the int ``-1`` rather than misreading it as
    an option flag -- and must round-trip as ``-1``."""
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(config_file, model_dir, "--worker-max-respawns", "-1")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert (
        config["models"]["limit-model"]["load_config"]["worker_max_respawns"] == -1
    )


def test_add_omits_worker_max_respawns_when_unspecified(tmp_path) -> None:
    """Untouched the option carries no value (None), so the supervisor keeps
    its own default budget of 2 -- the key must not be written at all."""
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = _invoke_add(config_file, model_dir)
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert "worker_max_respawns" not in config["models"]["limit-model"]["load_config"]


def test_add_does_not_save_vlm_type(tmp_path) -> None:
    config_file = tmp_path / "openarc_config.json"
    model_dir = _model_dir(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "add",
            "--model-name",
            "test-vlm",
            "--model-path",
            str(model_dir),
            "--engine",
            "ovgenai",
            "--model-type",
            "vlm",
            "--device",
            "CPU",
        ],
        env={"OPENARC_CONFIG_FILE": str(config_file)},
    )

    assert result.exit_code == 0
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    model_config = config["models"]["test-vlm"]["load_config"]
    assert "vlm_type" not in model_config
