import logging

from forkplan_images.cli import configure_logging, parser


def test_verbose_flag_sets_debug_and_default_is_warning():
    assert parser().parse_args(["-v"]).verbose and not parser().parse_args([]).verbose
    logger = logging.getLogger("forkplan_images")
    configure_logging(False)
    assert logger.level == logging.WARNING and not logger.propagate
    configure_logging(True)
    assert logger.level == logging.DEBUG


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
