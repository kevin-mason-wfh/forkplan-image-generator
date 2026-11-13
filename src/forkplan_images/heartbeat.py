"""Keeps an attempt's lease alive while slow work runs, and detects when ownership is lost."""
import logging
from collections.abc import Callable
from threading import Event, Thread
from typing import Any, TypeVar

from .repository import Repository
from .state import Attempt, LeaseLost

T = TypeVar("T")
log = logging.getLogger(__name__)


class Heartbeat:
    def __init__(self, repository: Repository, attempt: Attempt, seconds: int):
        self.repository, self.attempt, self.seconds = repository, attempt, seconds
        self.stop = Event()
        self.lost = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.seconds / 3):
            try:
                self.repository.renew(self.attempt.id, self.attempt.lease_token, self.seconds)
            except Exception:
                log.warning("lease renewal failed for attempt %s; stopping", self.attempt.id)
                self.lost.set()
                return

    def check(self) -> None:
        if self.lost.is_set():
            raise LeaseLost("Lease renewal failed; processing stopped")

    def fence(self) -> None:
        self.check()
        try:
            self.repository.renew(self.attempt.id, self.attempt.lease_token, self.seconds)
        except Exception:
            self.lost.set()
            raise LeaseLost("Lease renewal failed; processing stopped") from None
        self.check()

    def call(self, operation: Callable[[], T]) -> T:
        self.fence()
        try:
            result = operation()
        finally:
            self.check()
        self.check()
        return result

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join()
