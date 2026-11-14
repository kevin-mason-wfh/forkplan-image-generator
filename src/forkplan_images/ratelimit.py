"""Keeps requests under a per-minute limit by spacing them evenly."""
import logging
import time
from collections.abc import Callable

from .network import Checkpoint

log = logging.getLogger(__name__)

MAX_SLEEP_SLICE_SECONDS = 5.0  # between slices the checkpoint runs, so a held lease stays alive and a lost one is noticed


class RateLimiter:
    """At most `per_minute` calls in any 60 seconds. Even spacing is stricter than a sliding window and needs no
    history: each call waits until 60/per_minute seconds after the previous one. per_minute <= 0 means no limit."""

    def __init__(self, per_minute: int, name: str = "requests", *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.name, self.clock, self.sleep = name, clock, sleep
        self._next_allowed = 0.0

    def wait(self, checkpoint: Checkpoint | None = None) -> None:
        if not self.interval:
            return
        delay = self._next_allowed - self.clock()
        if delay > 1:
            log.info("pacing %s: waiting %.0fs to stay under the rate limit", self.name, delay)
        while (remaining := self._next_allowed - self.clock()) > 0:
            self.sleep(min(remaining, MAX_SLEEP_SLICE_SECONDS))
            if checkpoint:
                checkpoint()
        self._next_allowed = self.clock() + self.interval
