import re
from collections.abc import Iterable


def redact(value: str, secrets: Iterable[str] = ()) -> str:
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)\b(?:https?|postgres(?:ql)?(?:\+\w+)?)://[^\s<>\"']+", "[URL]", value)
    value = re.sub(r"(?i)\bBearer\s+\S+", "******", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value[:2000]
