import random
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlsplit

import httpx

T = TypeVar("T")
Checkpoint = Callable[[], None]

REQUEST_TIMEOUT_SECONDS = 60.0  # text, download, presign and upload calls
IMAGE_TIMEOUT_SECONDS = 180.0  # image generation is much slower


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


BLOCKING_STATUSES = (401, 403, 429)


def blocking_status(exc: BaseException) -> int | None:
    """401/403/429 from any HTTP or SDK error (or its cause). Rejected credentials or an exhausted rate limit would
    fail every following recipe the same way, so the run stops instead of paying for more work that cannot finish."""
    for error in (exc, exc.__cause__):
        response = getattr(error, "response", None)
        code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
        if code in BLOCKING_STATUSES:
            return code
    return None


def blocking_message(exc: BaseException, code: int) -> str:
    service = "OpenAI" if type(exc).__module__.startswith("openai") else "A service"
    hint = getattr(exc, "hint", "")
    if hint:  # e.g. a refused service-account login: say why
        return f"{exc} ({hint}). Check the COGNITO_* settings, then run --retry-failures."
    if code == 429:
        return (f"{service} rate limit or quota reached (HTTP 429). Saved work is kept. Wait a minute, lower "
                "OPENAI_IMAGE_RPM / OPENAI_TEXT_RPM, or check your quota, then run --retry-failures.")
    return (f"{service} rejected the credentials (HTTP {code}). Check the API token or key in your configuration. "
            "Saved work is kept: fix it, then run --retry-failures.")


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
        req.extensions["timeout"] = dict.fromkeys(("connect", "read", "write", "pool"), REQUEST_TIMEOUT_SECONDS)
        response = client.send(req, auth=None, follow_redirects=False)
        response.raise_for_status()
        return response

    def retryable(exc: Exception) -> bool:
        return isinstance(exc, httpx.TransportError) or (
            isinstance(exc, httpx.HTTPStatusError)
            and (exc.response.status_code == 429 or exc.response.status_code >= 500)
        )

    return retry_call(send, retryable, checkpoint=checkpoint, sleep=sleep)
