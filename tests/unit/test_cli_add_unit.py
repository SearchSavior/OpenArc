import json

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
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert config["models"]["limit-model"]["worker_line_limit"] == 512 * 1024


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
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert config["models"]["limit-model"]["worker_line_limit"] == expected


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
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert "worker_line_limit" not in config["models"]["limit-model"]


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
    config = json.loads(config_file.read_text(encoding="utf-8"))
    model_config = config["models"]["test-vlm"]
    assert "vlm_type" not in model_config
