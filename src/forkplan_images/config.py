import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.engine import make_url

from .auth import CognitoSettings
from .network import IMAGE_TIMEOUT_SECONDS, require_https

# Deliberately narrower than the API; expand only after checking official documentation.
TEXT_MODELS = frozenset({"gpt-4o-2024-08-06", "gpt-5.4-mini", "gpt-5.4-nano"})
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


COGNITO_REQUIRED = ("COGNITO_REGION", "COGNITO_CLIENT_ID", "COGNITO_USERNAME", "COGNITO_PASSWORD")


def cognito_settings(env: Mapping[str, str]) -> CognitoSettings | None:
    """A Cognito service-account login, or None when no COGNITO_* login setting is present."""
    if not any(env.get(name, "").strip() for name in COGNITO_REQUIRED):
        return None
    missing = [name for name in COGNITO_REQUIRED if not env.get(name, "").strip()]
    if missing:
        raise ValueError(f"{', '.join(missing)} required when using a Cognito login")
    region, client_id = env["COGNITO_REGION"].strip(), env["COGNITO_CLIENT_ID"].strip()
    if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", region):
        raise ValueError("COGNITO_REGION must look like eu-west-2")
    if any(c.isspace() for c in client_id):
        raise ValueError("COGNITO_CLIENT_ID must not contain whitespace")
    token_use = env.get("COGNITO_TOKEN_USE", "access").strip().lower() or "access"
    if token_use not in ("access", "id"):
        raise ValueError("COGNITO_TOKEN_USE must be access or id")
    return CognitoSettings(region, client_id, env["COGNITO_USERNAME"].strip(), env["COGNITO_PASSWORD"],
                           env.get("COGNITO_CLIENT_SECRET", "").strip(), token_use)


@dataclass(frozen=True)
class Config:
    database_url: str = field(repr=False)
    api_base_url: str = field(repr=False)
    api_token: str = field(repr=False)  # may be empty when a Cognito service account is configured
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
    cognito: CognitoSettings | None = None
    text_rpm: int = 0  # 0 = no client-side pacing
    image_rpm: int = 0

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
        cognito = cognito_settings(env)
        api_token = env.get("API_TOKEN", "").strip()
        if not cognito:  # with both set, the fixed token is tried first and the login is the backup
            api_token = required("API_TOKEN")
        for name, value in (("API_TOKEN", api_token), ("OPENAI_API_KEY", required("OPENAI_API_KEY"))):
            if any(c.isspace() for c in value):
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
        def per_minute(name: str) -> int:
            try:
                value = int(env.get(name, "0").strip() or "0")
            except ValueError:
                raise ValueError(f"{name} must be a whole number of requests per minute (0 = no limit)") from None
            if value < 0:
                raise ValueError(f"{name} must not be negative")
            return value

        try:
            seconds = int(env.get("LEASE_SECONDS", str(DEFAULT_LEASE_SECONDS)))
        except ValueError:
            raise ValueError("LEASE_SECONDS must be an integer") from None
        if seconds <= IMAGE_TIMEOUT_SECONDS:
            raise ValueError(f"LEASE_SECONDS must exceed the {IMAGE_TIMEOUT_SECONDS:.0f}-second image timeout")
        return cls(
            database_url=database_url, api_base_url=api_base_url, api_token=api_token, cognito=cognito,
            openai_api_key=required("OPENAI_API_KEY"), text_model=text_model, image_model=image_model,
            quality=quality, size=size, artifact_dir=Path(env.get("ARTIFACT_DIR", "artifacts")),
            lease_seconds=seconds, image_provider=provider, gemini_api_key=gemini_key,
            local_image_dir=optional_path("LOCAL_IMAGE_DIR"), cost_log_path=optional_path("COST_LOG_PATH"),
            text_rpm=per_minute("OPENAI_TEXT_RPM"), image_rpm=per_minute("OPENAI_IMAGE_RPM"))

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
            self.gemini_api_key, *((self.cognito.password, self.cognito.client_secret) if self.cognito else ()),
        ) if value)
