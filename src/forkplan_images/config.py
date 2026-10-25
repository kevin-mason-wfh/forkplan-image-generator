from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from sqlalchemy.engine import make_url

from .generation import IMAGE_TIMEOUT
from .network import require_https


# Deliberately narrower than the API; expand only after checking official documentation.
TEXT_MODELS = frozenset({"gpt-4o-2024-08-06"})
IMAGE_MODELS = frozenset({"gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"})
IMAGE_QUALITIES = frozenset({"low", "medium", "high", "auto"})
IMAGE_SIZES = frozenset({"1024x1024"})
DEFAULT_QUALITY = "medium"
DEFAULT_SIZE = "1024x1024"
DEFAULT_LEASE_SECONDS = 300


@dataclass(frozen=True)
class Config:
    database_url: str = field(repr=False)
    api_base_url: str = field(repr=False)
    api_token: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    text_model: str
    image_model: str
    quality: str
    size: str
    artifact_dir: Path
    lease_seconds: int

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "Config":
        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value or value.startswith("<"):
                raise ValueError(f"{name} is required")
            return value

        database_url = required("DATABASE_URL")
        try:
            url = make_url(database_url)
            if url.drivername not in ("postgresql", "postgresql+psycopg"):
                raise ValueError()
        except Exception:
            raise ValueError("DATABASE_URL must be a Postgres URL") from None
        api_base_url = required("API_BASE_URL")
        require_https(api_base_url)
        for name in ("API_TOKEN", "OPENAI_API_KEY"):
            if any(c.isspace() for c in required(name)):
                raise ValueError(f"{name} must not contain whitespace")
        text_model, image_model = required("OPENAI_TEXT_MODEL"), required("OPENAI_IMAGE_MODEL")
        quality = env.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY)
        size = env.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE)
        for name, value, allowed in (
            ("OPENAI_TEXT_MODEL", text_model, TEXT_MODELS),
            ("OPENAI_IMAGE_MODEL", image_model, IMAGE_MODELS),
            ("OPENAI_IMAGE_QUALITY", quality, IMAGE_QUALITIES),
            ("OPENAI_IMAGE_SIZE", size, IMAGE_SIZES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} unsupported; choose from {', '.join(sorted(allowed))}")
        try:
            seconds = int(env.get("LEASE_SECONDS", str(DEFAULT_LEASE_SECONDS)))
        except ValueError:
            raise ValueError("LEASE_SECONDS must be an integer") from None
        if seconds <= IMAGE_TIMEOUT:
            raise ValueError("LEASE_SECONDS must exceed the 180-second image timeout")
        return cls(database_url, api_base_url, required("API_TOKEN"), required("OPENAI_API_KEY"),
                   text_model, image_model, quality, size,
                   Path(env.get("ARTIFACT_DIR", "artifacts")), seconds)

    @property
    def settings(self) -> dict[str, str]:
        return {"text_model": self.text_model, "image_model": self.image_model,
                "quality": self.quality, "size": self.size, "output_format": "png"}

    @property
    def secrets(self) -> tuple[str, ...]:
        url = make_url(self.database_url)
        return tuple(value for value in (
            self.database_url, url.password, url.username, self.api_token, self.openai_api_key
        ) if value)
