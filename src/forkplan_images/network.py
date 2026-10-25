import random
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlsplit

import httpx


T = TypeVar("T")
Checkpoint = Callable[[], None]


def retry_call(operation: Callable[[], T], retryable: Callable[[Exception], bool], *,
               checkpoint: Checkpoint | None = None,
               sleep: Callable[[float], None] = time.sleep) -> T:
    for attempt in range(3):
        if checkpoint:
            checkpoint()
        try:
            return operation()
        except Exception as exc:
            if attempt == 2 or not retryable(exc):
                raise
            if checkpoint:
                checkpoint()
            sleep(2 ** attempt + random.uniform(0, 0.5))
    raise AssertionError("unreachable")


def require_https(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Network URL must use HTTPS without embedded credentials")
    return url


def request(client: httpx.Client, method: str, url: str, *,
            headers: dict[str, str] | None = None, content: bytes | None = None,
            checkpoint: Checkpoint | None = None,
            sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    require_https(url)

    def send() -> httpx.Response:
        # Explicit Request avoids inheriting client-wide auth/cookies/default headers.
        req = httpx.Request(method, url, headers=headers, content=content)
        req.extensions["timeout"] = {key: 60.0 for key in ("connect", "read", "write", "pool")}
        response = client.send(req, auth=None, follow_redirects=False)
        response.raise_for_status()
        return response

    def retryable(exc: Exception) -> bool:
        return isinstance(exc, httpx.TransportError) or (
            isinstance(exc, httpx.HTTPStatusError)
            and (exc.response.status_code == 429 or exc.response.status_code >= 500)
        )

    return retry_call(send, retryable, checkpoint=checkpoint, sleep=sleep)
