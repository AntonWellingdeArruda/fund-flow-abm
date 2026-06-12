import pytest

from fund_flow import config


def test_load_dotenv_populates(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_KEY=abc123\nBAR=hello world\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.setattr(config, "_loaded", False)
    config.load_dotenv(env)
    assert config.get_secret("FOO_KEY") == "abc123"
    assert config.get_secret("BAR") == "hello world"


def test_existing_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=fromfile\n")
    monkeypatch.setenv("FOO_KEY", "fromenv")
    monkeypatch.setattr(config, "_loaded", False)
    config.load_dotenv(env)
    assert config.get_secret("FOO_KEY") == "fromenv"   # setdefault keeps env


def test_missing_secret_raises(monkeypatch):
    monkeypatch.setattr(config, "_loaded", True)       # skip file load
    monkeypatch.delenv("DEFINITELY_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="Missing secret"):
        config.get_secret("DEFINITELY_MISSING")
