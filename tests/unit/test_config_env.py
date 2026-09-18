"""Tests for .env export and the provider-key pre-flight check."""

import pytest
import typer

from prompt_opt import config
from prompt_opt.cli.main import _require_api_key


@pytest.fixture
def clean_keys(monkeypatch, tmp_path):
    """Unset provider keys (restored on teardown, even if load_dotenv sets them)."""
    for var in config.PROVIDER_KEY_VARS:
        monkeypatch.setenv(var, "placeholder")  # registers teardown restore
        monkeypatch.delenv(var)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_key_in_dotenv_is_visible_to_preflight(clean_keys):
    (clean_keys / ".env").write_text("OPENAI_API_KEY=sk-real-key\n", encoding="utf-8")
    assert not config.has_provider_key()

    config.load_env_file()

    assert config.has_provider_key()
    _require_api_key("optimize")  # must not raise


def test_shell_environment_wins_over_dotenv(clean_keys, monkeypatch):
    (clean_keys / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")

    config.load_env_file()

    import os

    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def test_placeholder_values_do_not_count(clean_keys):
    (clean_keys / ".env").write_text(
        "OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\nGOOGLE_API_KEY=...\n",
        encoding="utf-8",
    )
    config.load_env_file()

    assert not config.has_provider_key()
    with pytest.raises(typer.Exit):
        _require_api_key("optimize")


def test_missing_dotenv_is_fine(clean_keys):
    config.load_env_file()  # no .env in cwd: no error
    assert not config.has_provider_key()
