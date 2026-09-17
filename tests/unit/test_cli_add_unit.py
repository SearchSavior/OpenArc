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


def test_add_does_not_save_vlm_type(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
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
    entry = config["models"]["test-vlm"]
    assert "load_config" in entry
    assert "vlm_type" not in entry["load_config"]


def test_add_writes_nested_shape_with_sampler_config(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    model_dir = _model_dir(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "add",
            "--model-name",
            "test-llm",
            "--model-path",
            str(model_dir),
            "--engine",
            "ovgenai",
            "--model-type",
            "llm",
            "--device",
            "CPU",
            "--sampler-config",
            '{"temperature": 0.7, "top_k": 40}',
            "--tool-call-parser",
            "hermes",
        ],
        env={"OPENARC_CONFIG_FILE": str(config_file)},
    )

    assert result.exit_code == 0
    entry = yaml.safe_load(config_file.read_text(encoding="utf-8"))["models"]["test-llm"]
    # load fields nest under load_config; request defaults are siblings.
    assert entry["load_config"]["tool_call_parser"] == "hermes"
    assert entry["load_config"]["model_type"] == "llm"
    assert entry["sampler_config"] == {"temperature": 0.7, "top_k": 40}


def test_add_rejects_invalid_sampler_config(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    model_dir = _model_dir(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "add",
            "--model-name",
            "bad",
            "--model-path",
            str(model_dir),
            "--engine",
            "ovgenai",
            "--model-type",
            "llm",
            "--device",
            "CPU",
            "--sampler-config",
            '{"temperature": "not-a-number"}',
        ],
        env={"OPENARC_CONFIG_FILE": str(config_file)},
    )

    assert result.exit_code != 0
    assert not config_file.exists()
