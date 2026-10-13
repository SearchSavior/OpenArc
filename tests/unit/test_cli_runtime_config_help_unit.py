"""Public CLI help should expose the advanced runtime configuration reference."""

import re

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.mark.parametrize("width", [80, 160])
def test_add_help_links_runtime_configuration(width):
    result = CliRunner().invoke(cli, ["add", "--help"], terminal_width=width)
    assert result.exit_code == 0, result.output
    assert "--runtime-config" in result.output
    # Rich wraps long links in narrow terminals. Ignore presentation whitespace.
    text = re.sub(r"\s+", "", result.output.replace("│", ""))
    assert "https://searchsavior.github.io/OpenArc/configure/#runtime_config" in text
