import random
import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx


def require_https(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Network URL must use HTTPS without embedded credentials")
    return url


def request(client: httpx.Client, method: str, url: str, *,
            headers: dict[str, str] | None = None, content: bytes | None = None,
            sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    require_https(url)
    for attempt in range(3):
        try:
            # Explicit Request avoids inheriting client-wide auth/cookies/default headers.
            req = httpx.Request(method, url, headers=headers, content=content)
            req.extensions["timeout"] = {key: 60.0 for key in ("connect", "read", "write", "pool")}
            response = client.send(req, auth=None, follow_redirects=False)
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status != 429 and status < 500:
                    raise
            if attempt == 2:
                raise
            sleep(2 ** attempt + random.uniform(0, 0.5))
    raise AssertionError("unreachable")
