"""Unit tests for `openarc serve start --session-id-header/--sih`.

The flag names the HTTP header that carries a client session id. When set,
`serve start` must export ``OPENARC_SESSION_ID_HEADER`` to that name and hand it
to ``start_server`` (which enables the server-side session registry). When it is
absent, the env var must be exported as empty (sessions OFF) so behaviour is
unchanged. ``start_server`` is monkeypatched so nothing is actually launched.
"""

import os

import pytest  # type: ignore[import]
from click.testing import CliRunner

from src.cli import cli
import src.cli.modules.launch_server as launch_server


def _make_record(monkeypatch: "pytest.MonkeyPatch") -> dict:
    """Replace start_server with a recorder that captures what it was handed,
    including the env as seen at call time (the CLI edits os.environ before
    calling it). Returns a mutable dict the test inspects after the call."""
    record: dict = {}

    def fake_start_server(host=None, port=None, reload=False, verbose=0, session_id_header=""):
        record["host"] = host
        record["port"] = port
        record["session_id_header"] = session_id_header
        record["env_session_header"] = os.environ.get("OPENARC_SESSION_ID_HEADER")

    monkeypatch.setattr(launch_server, "start_server", fake_start_server)
    return record


def test_sih_sets_env_and_threads_it(
    monkeypatch: "pytest.MonkeyPatch", tmp_path
) -> None:
    cfg = tmp_path / "openarc_config.yaml"
    cfg.write_text("")
    record = _make_record(monkeypatch)

    result = CliRunner().invoke(
        cli,
        ["serve", "start", "--sih", "X-Session-Id"],
        env={"OPENARC_CONFIG_FILE": str(cfg)},
    )
    assert result.exit_code == 0, result.output
    assert record["session_id_header"] == "X-Session-Id"
    assert record["env_session_header"] == "X-Session-Id"
    assert "OPENARC_SESSION_ID_HEADER=X-Session-Id" in result.output


def test_no_sih_exports_empty_and_disables(
    monkeypatch: "pytest.MonkeyPatch", tmp_path
) -> None:
    cfg = tmp_path / "openarc_config.yaml"
    cfg.write_text("")
    record = _make_record(monkeypatch)

    result = CliRunner().invoke(
        cli,
        ["serve", "start"],
        env={"OPENARC_CONFIG_FILE": str(cfg)},
    )
    assert result.exit_code == 0, result.output
    # An explicit empty is what makes "unset" deterministic (not an inherited value).
    assert record["session_id_header"] in ("", None)
    assert record["env_session_header"] == ""


def test_sih_help_is_shown() -> None:
    result = CliRunner().invoke(cli, ["serve", "start", "--help"])
    assert result.exit_code == 0, result.output
    assert "--session-id-header" in result.output
    assert "--sih" in result.output
