from unittest.mock import Mock
from uuid import uuid4

import pytest

from forkplan_images import cli
from forkplan_images.config import Config
from forkplan_images.orchestrator import Options, Result, Summary


def environment():
    return {
        "DATABASE_URL": "postgresql+psycopg://" + "db.example.test/catalog",
        "API_BASE_URL": "https://api.example.test", "API_TOKEN": "placeholder-api",
        "OPENAI_API_KEY": "placeholder-openai", "OPENAI_TEXT_MODEL": "gpt-4o-2024-08-06",
        "OPENAI_IMAGE_MODEL": "gpt-image-1.5",
    }


def test_configuration_models_required_and_settings_safe():
    env = environment()
    config = Config.from_environment(env)
    assert config.settings["size"] == "1024x1024"
    assert config.lease_seconds > 180
    assert "placeholder-api" not in repr(config)
    assert "placeholder-openai" not in repr(config)
    assert "placeholder" not in str(config.settings)
    for key in ("OPENAI_TEXT_MODEL", "OPENAI_IMAGE_MODEL"):
        with pytest.raises(ValueError, match="required"):
            Config.from_environment({k: v for k, v in env.items() if k != key})


@pytest.mark.parametrize("key,value", [
    ("OPENAI_IMAGE_MODEL", "unknown"), ("OPENAI_TEXT_MODEL", "unknown"),
    ("OPENAI_IMAGE_SIZE", "1536x1024"), ("OPENAI_IMAGE_QUALITY", "ultra"),
    ("LEASE_SECONDS", "180"), ("LEASE_SECONDS", "invalid"),
    ("API_BASE_URL", "http://api.example.test"), ("DATABASE_URL", "sqlite://"),
])
def test_unsupported_configuration_fails_fast(key, value):
    with pytest.raises(ValueError):
        Config.from_environment(environment() | {key: value})


@pytest.mark.parametrize("flags", [
    ["--limit", "-1"], ["--max-images", "-1"],
    ["--redo", str(uuid4()), "--dry-run"],
    ["--redo", str(uuid4()), "--ids", str(uuid4())], ["--note", "reason"],
])
def test_usage_conflicts_before_execution(monkeypatch, capsys, flags):
    execute = Mock()
    monkeypatch.setattr(cli, "execute", execute)
    assert cli.main(flags) == 2
    execute.assert_not_called()
    assert "error" in capsys.readouterr().err


def test_cli_summary_redacted_and_exit_codes(monkeypatch, capsys):
    config = Config.from_environment(environment())
    monkeypatch.setattr(cli.Config, "from_environment", lambda _: config)
    recipe_id = uuid4()
    monkeypatch.setattr(cli, "execute", lambda *_: Summary(
        [Result(recipe_id, "generation_uncertain",
                "placeholder-api https://bucket.example.test/?signature=placeholder")], 1))
    assert cli.main(["--ids", str(recipe_id)]) == 1
    output = capsys.readouterr().out
    assert "placeholder-api" not in output
    assert "https://" not in output
    assert "uncertain: 1" in output
    monkeypatch.setattr(cli, "execute", lambda *_: Summary())
    assert cli.main([]) == 0


def test_help_explains_cost_and_caps(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "text-model calls may incur cost" in help_text
    assert "recovered originals" in help_text
    assert "resumed attempts" in help_text


def test_argument_uuid_errors():
    with pytest.raises(SystemExit) as error:
        cli.main(["--ids", "not-a-uuid"])
    assert error.value.code == 2
