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
