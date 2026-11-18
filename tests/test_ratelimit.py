import pytest
from test_cli_config import environment
from test_orchestrator import _http_error, setup

from forkplan_images.config import Config
from forkplan_images.orchestrator import Options
from forkplan_images.ratelimit import RateLimiter
from forkplan_images.state import Stage


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def limiter(per_minute, clock):
    return RateLimiter(per_minute, "test", clock=clock, sleep=clock.sleep)


def test_calls_are_spaced_evenly_and_never_exceed_the_limit_in_any_minute():
    clock = FakeClock()
    pacer, times = limiter(10, clock), []
    for _ in range(30):
        pacer.wait()
        times.append(clock.now)
    assert all(b - a >= 6.0 - 1e-9 for a, b in zip(times, times[1:], strict=False))  # 10 per minute = one every 6 seconds
    assert all(sum(1 for t in times if start <= t < start + 60) <= 10 for start in times)


def test_slow_work_is_not_delayed_and_zero_means_unlimited():
    clock = FakeClock()
    pacer = limiter(10, clock)
    pacer.wait()
    clock.now += 20  # the real work took longer than the interval
    pacer.wait()
    assert clock.sleeps == []
    unlimited = limiter(0, clock)
    for _ in range(100):
        unlimited.wait()
    assert clock.sleeps == []


def test_long_waits_run_the_checkpoint_so_the_lease_stays_alive():
    clock = FakeClock()
    pacer, checks = limiter(1, clock), []
    pacer.wait()
    pacer.wait(lambda: checks.append(clock.now))  # a 60 second wait
    assert len(checks) >= 12 and max(clock.sleeps) <= 5.0


def test_a_lost_lease_during_a_wait_stops_waiting():
    clock = FakeClock()
    pacer = limiter(1, clock)
    pacer.wait()

    def lost():
        raise RuntimeError("lease lost")
    with pytest.raises(RuntimeError):
        pacer.wait(lost)


@pytest.mark.parametrize("value,expected", [("", 0), ("12", 12), ("0", 0)])
def test_rpm_settings_parse(value, expected):
    config = Config.from_environment({**environment(), "OPENAI_IMAGE_RPM": value, "OPENAI_TEXT_RPM": value})
    assert config.image_rpm == expected and config.text_rpm == expected


@pytest.mark.parametrize("value", ["fast", "-1", "2.5"])
def test_bad_rpm_settings_are_rejected(value):
    with pytest.raises(ValueError, match="OPENAI_IMAGE_RPM"):
        Config.from_environment({**environment(), "OPENAI_IMAGE_RPM": value})


def test_image_requests_are_paced_before_the_generating_marker(tmp_path):
    runner, repo, _, _, generate, _, _ = setup(tmp_path, 2)
    events = []

    class Recording:
        def wait(self, checkpoint=None):
            assert all(a.stage != Stage.GENERATING for a in repo.attempts.values())  # nothing marked as dispatched yet
            events.append("wait")

    runner.image_limiter = Recording()
    generate.generate.side_effect = lambda *a, **k: events.append("generate") or generate.generate.return_value
    runner.run(Options(limit=2, max_images=2))
    assert events == ["wait", "generate", "wait", "generate"]


@pytest.mark.parametrize("where", ["image", "text"])
def test_a_429_stops_the_run_and_keeps_saved_work(tmp_path, where):
    runner, repo, _, describe, generate, _, _ = setup(tmp_path, 3)
    if where == "image":
        from forkplan_images.generation import GenerationError
        error = GenerationError("rate limited", False)
        error.__cause__ = _http_error(429)
        generate.generate.side_effect = error
    else:
        describe.describe.side_effect = _http_error(429)
    result = runner.run(Options(limit=3, max_images=3))
    assert len(result.results) == 1 and result.exit_code == 1
    assert "rate limit" in result.stopped_because and "--retry-failures" in result.stopped_because
    assert len(repo.attempts) == 1
