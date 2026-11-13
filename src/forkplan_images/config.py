import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.engine import make_url

from .network import IMAGE_TIMEOUT_SECONDS, require_https

# Deliberately narrower than the API; expand only after checking official documentation.
TEXT_MODELS = frozenset({"gpt-4o-2024-08-06"})
# Any GPT Image family ID (e.g. gpt-image-1.5, gpt-image-2, dated snapshots) is accepted so newer
# models need no code change. gpt-image-1 is closed upstream; the API will reject it if used.
IMAGE_MODEL_PATTERN = re.compile(r"gpt-image-[A-Za-z0-9][A-Za-z0-9._-]*")
IMAGE_QUALITIES = frozenset({"low", "medium", "high", "auto"})
IMAGE_SIZES = frozenset({"1024x1024"})
GEMINI_IMAGE_MODEL_PATTERN = re.compile(r"gemini-[A-Za-z0-9._-]*image[A-Za-z0-9._-]*")
IMAGE_PROVIDERS = frozenset({"openai", "gemini"})
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
    image_provider: str = "openai"
    gemini_api_key: str = field(default="", repr=False)
    local_image_dir: Path | None = None
    cost_log_path: Path | None = None

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "Config":
        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value or value.startswith("<"):
                raise ValueError(f"{name} is required")
            return value

        def optional_path(name: str) -> Path | None:
            value = env.get(name, "").strip()
            return Path(value) if value else None

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
        provider = env.get("IMAGE_PROVIDER", "openai").strip() or "openai"
        if provider not in IMAGE_PROVIDERS:
            raise ValueError(f"IMAGE_PROVIDER unsupported; choose from {', '.join(sorted(IMAGE_PROVIDERS))}")
        gemini_key = ""
        text_model = required("OPENAI_TEXT_MODEL")
        if provider == "gemini":
            # Text descriptions still use OpenAI; only the image dispatch switches provider.
            gemini_key = required("GEMINI_API_KEY")
            if any(c.isspace() for c in gemini_key):
                raise ValueError("GEMINI_API_KEY must not contain whitespace")
            image_model = required("GEMINI_IMAGE_MODEL")
            if not GEMINI_IMAGE_MODEL_PATTERN.fullmatch(image_model):
                raise ValueError("GEMINI_IMAGE_MODEL unsupported; expected a gemini-*image* model ID")
        else:
            image_model = required("OPENAI_IMAGE_MODEL")
        quality = env.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY)
        size = env.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE)
        for name, value, allowed in (
            ("OPENAI_TEXT_MODEL", text_model, TEXT_MODELS),
            ("OPENAI_IMAGE_QUALITY", quality, IMAGE_QUALITIES),
            ("OPENAI_IMAGE_SIZE", size, IMAGE_SIZES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} unsupported; choose from {', '.join(sorted(allowed))}")
        if provider == "openai" and not IMAGE_MODEL_PATTERN.fullmatch(image_model):
            raise ValueError("OPENAI_IMAGE_MODEL unsupported; expected a gpt-image-* model ID")
        try:
            seconds = int(env.get("LEASE_SECONDS", str(DEFAULT_LEASE_SECONDS)))
        except ValueError:
            raise ValueError("LEASE_SECONDS must be an integer") from None
        if seconds <= IMAGE_TIMEOUT_SECONDS:
            raise ValueError(f"LEASE_SECONDS must exceed the {IMAGE_TIMEOUT_SECONDS:.0f}-second image timeout")
        return cls(
            database_url=database_url, api_base_url=api_base_url, api_token=required("API_TOKEN"),
            openai_api_key=required("OPENAI_API_KEY"), text_model=text_model, image_model=image_model,
            quality=quality, size=size, artifact_dir=Path(env.get("ARTIFACT_DIR", "artifacts")),
            lease_seconds=seconds, image_provider=provider, gemini_api_key=gemini_key,
            local_image_dir=optional_path("LOCAL_IMAGE_DIR"), cost_log_path=optional_path("COST_LOG_PATH"))

    @property
    def cost_log(self) -> Path:
        return self.cost_log_path or self.artifact_dir / "cost_log.csv"

    @property
    def settings(self) -> dict[str, str]:
        base = {"text_model": self.text_model, "image_model": self.image_model,
                "quality": self.quality, "size": self.size, "output_format": "png"}
        return {**base, "image_provider": self.image_provider} if self.image_provider != "openai" else base

    @property
    def secrets(self) -> tuple[str, ...]:
        url = make_url(self.database_url)
        return tuple(value for value in (
            self.database_url, url.password, url.username, self.api_token, self.openai_api_key,
            self.gemini_api_key,
        ) if value)
