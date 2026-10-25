from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class Stage(StrEnum):
    SELECTED = "selected"
    DESCRIBED = "described"
    PROMPT_BUILT = "prompt_built"
    GENERATING = "generating"
    GENERATED = "generated"
    PROCESSED = "processed"
    PRESIGNED = "presigned"
    UPLOADED = "uploaded"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "generation_uncertain"
    RETIRED = "retired"


ACTIVE = tuple(Stage)[:8]
SUCCESSFUL = tuple(s for s in tuple(Stage)[:9] if s != Stage.GENERATING)


class LeaseLost(RuntimeError):
    pass


class ClaimConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class Recipe:
    id: UUID
    name: str
    category: str | None = None
    instructions: str | None = None
    ingredients: tuple[dict[str, Any], ...] = ()
    image_key: str | None = None
    status: str = "PUBLISHED"


@dataclass
class Attempt:
    resource_id: UUID
    id: UUID = field(default_factory=uuid4)
    resource_type: str = "recipe"
    attempt_number: int = 1
    stage: Stage = Stage.SELECTED
    last_successful_stage: Stage = Stage.SELECTED
    description: str | None = None
    final_prompt: str | None = None
    prompt_version: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    original_key: str | None = None
    jpeg_key: str | None = None
    jpeg_sha256: str | None = None
    jpeg_length: int | None = None
    image_key: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    superseded_by: UUID | None = None
    redo_note: str | None = None
    lease_owner: str | None = None
    lease_token: UUID | None = None
    lease_expires_at: Any = None

    @property
    def original_storage_key(self) -> str:
        return f"attempts/{self.id}/original"

    @property
    def jpeg_storage_key(self) -> str:
        return f"attempts/{self.id}/image.jpg"
