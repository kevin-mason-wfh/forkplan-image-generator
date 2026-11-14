import logging

import pytest

from forkplan_images.cli import configure_logging, main, parser, resolve_log_level


def test_log_level_precedence_default_and_validation():
    assert resolve_log_level(False, None, {}) == "info"  # progress is visible by default
    assert resolve_log_level(False, None, {"LOG_LEVEL": "WARNING"}) == "warning"
    assert resolve_log_level(False, "error", {"LOG_LEVEL": "debug"}) == "error"  # the option beats the environment
    assert resolve_log_level(True, "error", {"LOG_LEVEL": "warning"}) == "debug"  # -v beats both
    with pytest.raises(ValueError):
        resolve_log_level(False, None, {"LOG_LEVEL": "chatty"})
    assert parser().parse_args(["--log-level", "DEBUG"]).log_level == "debug"
    with pytest.raises(SystemExit):
        parser().parse_args(["--log-level", "loud"])


def test_configure_logging_applies_the_level():
    logger = logging.getLogger("forkplan_images")
    for name, number in (("debug", logging.DEBUG), ("info", logging.INFO), ("warning", logging.WARNING),
                         ("error", logging.ERROR)):
        configure_logging(name)
        assert logger.level == number and not logger.propagate


def test_bad_log_level_in_the_environment_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "chatty")
    assert main([]) == 2
    assert "LOG_LEVEL" in capsys.readouterr().err


def test_progress_is_logged_at_info_and_hidden_at_warning(tmp_path, caplog):
    from test_orchestrator import setup

    from forkplan_images.orchestrator import Options

    logger = logging.getLogger("forkplan_images")
    logger.propagate = True
    try:
        runner, *_ = setup(tmp_path)
        with caplog.at_level(logging.INFO, logger="forkplan_images"):
            runner.run(Options())
        text = caplog.text
        for expected in ("Run started", "Selected 1 recipe(s)", "[1/1]", "described", "requesting the image", "completed",
                         "Run finished"):
            assert expected in text
        caplog.clear()
        runner2, *_ = setup(tmp_path / "second")
        with caplog.at_level(logging.WARNING, logger="forkplan_images"):
            runner2.run(Options())
        assert caplog.text == ""  # a clean run is silent at warning level
    finally:
        logger.propagate = False


def test_failure_logs_are_redacted(tmp_path, caplog):
    from test_orchestrator import setup

    from forkplan_images.generation import GenerationError
    from forkplan_images.orchestrator import Options

    runner, _, _, _, generate, _, _ = setup(tmp_path)
    generate.generate.side_effect = GenerationError("bad key sk-abcdef123456", False)
    logger = logging.getLogger("forkplan_images")
    logger.propagate = True  # caplog attaches to the root logger
    with caplog.at_level(logging.INFO, logger="forkplan_images"):
        runner.run(Options())
    logger.propagate = False
    assert "sk-abcdef123456" not in caplog.text and "failed" in caplog.text
