from dataclasses import dataclass, fields
from typing import Any, Protocol
from uuid import UUID, uuid4
import json

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine

from .redaction import redact
from .state import ACTIVE, Attempt, ClaimConflict, LeaseLost, Recipe, Stage


@dataclass(frozen=True)
class Work:
    recipe: Recipe
    attempt: Attempt | None = None


class Repository(Protocol):
    def candidates(self, limit: int, ids: tuple[UUID, ...], retry: bool) -> list[Work]: ...
    def claim(self, work: Work, owner: str, seconds: int) -> Attempt: ...
    def redo(self, resource_id: UUID, owner: str, seconds: int, note: str | None) -> Work: ...
    def update(self, attempt_id: UUID, token: UUID, **changes: Any) -> Attempt: ...
    def renew(self, attempt_id: UUID, token: UUID, seconds: int) -> None: ...
    def release(self, attempt_id: UUID, token: UUID) -> None: ...


MUTABLE = {
    "stage", "last_successful_stage", "description", "final_prompt", "prompt_version",
    "settings", "original_key", "jpeg_key", "jpeg_sha256", "jpeg_length", "image_key",
    "error_type", "error_message",
}
ATTEMPT_FIELDS = {f.name for f in fields(Attempt)}


def clean_changes(changes: dict[str, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    if not changes or not changes.keys() <= MUTABLE:
        raise ValueError("Unsupported attempt update")
    return {
        key: redact(str(value), secrets) if key.startswith("error_") and value else value
        for key, value in changes.items()
    }


def decode(row: Any) -> Attempt:
    values = {k: v for k, v in row.items() if k in ATTEMPT_FIELDS}
    values["stage"] = Stage(values["stage"])
    values["last_successful_stage"] = Stage(values["last_successful_stage"])
    return Attempt(**values)


class PostgresRepository:
    """Transactions commit before returning; all mutation predicates fence expired tokens."""

    def __init__(self, engine: Engine, secrets: tuple[str, ...] = ()):
        self.engine = engine
        self.secrets = secrets

    def _recipe(self, conn: Connection, resource_id: UUID) -> Recipe:
        row = conn.execute(text("SELECT * FROM catalog_recipes WHERE id = :id"),
                           {"id": resource_id}).mappings().one()
        ingredients = conn.execute(text("""
            SELECT i.name, ri.quantity, ri.unit FROM catalog_recipe_ingredients ri
            JOIN catalog_ingredients i ON i.id = ri.catalog_ingredient_id
            WHERE ri.catalog_recipe_id = :id ORDER BY i.id
        """), {"id": resource_id}).mappings()
        return Recipe(
            id=row["id"], name=row["name"], category=row["category"],
            instructions=row["instructions"], image_key=row["image_key"],
            status=row["status"], ingredients=tuple(dict(i) for i in ingredients),
        )

    def candidates(self, limit: int, ids: tuple[UUID, ...] = (), retry: bool = False) -> list[Work]:
        stages = [s.value for s in ACTIVE] + ([Stage.FAILED.value] if retry else [])
        with self.engine.connect() as conn:
            attempts = conn.execute(text("""
                SELECT * FROM catalog_image_generations
                WHERE resource_type = 'recipe' AND stage IN :stages
                  AND (lease_expires_at IS NULL OR lease_expires_at <= now())
                  AND (:all_ids OR resource_id IN :ids)
                ORDER BY created_at, id LIMIT :limit
            """).bindparams(bindparam("stages", expanding=True), bindparam("ids", expanding=True)),
                {"stages": stages, "ids": ids, "all_ids": not ids, "limit": limit}).mappings()
            work = [Work(self._recipe(conn, row["resource_id"]), decode(row)) for row in attempts]
            remaining = limit - len(work)
            if remaining:
                rows = conn.execute(text("""
                    SELECT r.id FROM catalog_recipes r
                    WHERE (r.image_key IS NULL OR r.image_key = '')
                      AND r.status IN ('PUBLISHED', 'DRAFT')
                      AND (:all_ids OR r.id IN :ids)
                      AND NOT EXISTS (
                        SELECT 1 FROM catalog_image_generations g
                        WHERE g.resource_type = 'recipe' AND g.resource_id = r.id
                          AND g.stage <> 'retired')
                    ORDER BY r.id LIMIT :limit
                """).bindparams(bindparam("ids", expanding=True)),
                    {"ids": ids, "all_ids": not ids, "limit": remaining}).mappings()
                work.extend(Work(self._recipe(conn, row["id"])) for row in rows)
            return work

    def _lock_resource(self, conn: Connection, resource_id: UUID) -> None:
        # Serialize even first attempts, without ever locking/writing a catalog row.
        locked = conn.execute(text(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"
        ), {"key": f"recipe:{resource_id}"}).scalar_one()
        if not locked:
            raise ClaimConflict("Resource is being claimed")

    def _lock_attempt(self, conn: Connection, attempt_id: UUID) -> None:
        # Lock only; ownership is still checked atomically by the following UPDATE.
        conn.execute(text("SELECT id FROM catalog_image_generations WHERE id = :id FOR UPDATE"),
                     {"id": attempt_id})

    def _insert(self, conn: Connection, resource_id: UUID, number: int, owner: str,
                seconds: int, note: str | None = None, attempt_id: UUID | None = None) -> Attempt:
        row = conn.execute(text("""
            INSERT INTO catalog_image_generations
                (id, resource_id, resource_type, attempt_number, stage,
                 last_successful_stage, lease_owner, lease_token, lease_expires_at, redo_note)
            VALUES (:id, :resource_id, 'recipe', :number, 'selected', 'selected',
                    :owner, :token, clock_timestamp() + :seconds * interval '1 second', :note)
            RETURNING *
        """), {"id": attempt_id or uuid4(), "resource_id": resource_id, "number": number,
               "owner": owner, "token": uuid4(), "seconds": seconds,
               "note": redact(note, self.secrets) if note else None}).mappings().one()
        return decode(row)

    def claim(self, work: Work, owner: str, seconds: int) -> Attempt:
        with self.engine.begin() as conn:
            self._lock_resource(conn, work.recipe.id)
            if work.attempt:
                self._lock_attempt(conn, work.attempt.id)
                row = conn.execute(text("""
                    UPDATE catalog_image_generations SET
                        lease_owner = :owner, lease_token = :token,
                        lease_expires_at = clock_timestamp() + :seconds * interval '1 second',
                        stage = CASE WHEN stage = 'failed' THEN last_successful_stage ELSE stage END,
                        retry_count = retry_count + CASE WHEN stage = 'failed' THEN 1 ELSE 0 END,
                        error_type = NULL, error_message = NULL, updated_at = now()
                    WHERE id = :id AND stage = :expected
                      AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
                      AND stage IN ('selected','described','prompt_built','generating',
                                    'generated','processed','presigned','uploaded','failed')
                    RETURNING *
                """), {"owner": owner, "token": uuid4(), "seconds": seconds,
                       "id": work.attempt.id, "expected": work.attempt.stage.value}).mappings().first()
                if row is None:
                    raise ClaimConflict("Attempt already claimed or changed")
                return decode(row)
            exists = conn.execute(text("""
                SELECT 1 FROM catalog_image_generations
                WHERE resource_type = 'recipe' AND resource_id = :id AND stage <> 'retired'
            """), {"id": work.recipe.id}).first()
            recipe = self._recipe(conn, work.recipe.id)
            if exists or recipe.image_key or recipe.status not in ("PUBLISHED", "DRAFT"):
                raise ClaimConflict("Recipe is no longer fresh")
            number = conn.execute(text("""
                SELECT COALESCE(max(attempt_number), 0) + 1 FROM catalog_image_generations
                WHERE resource_type = 'recipe' AND resource_id = :id
            """), {"id": recipe.id}).scalar_one()
            return self._insert(conn, recipe.id, number, owner, seconds)

    def redo(self, resource_id: UUID, owner: str, seconds: int, note: str | None) -> Work:
        with self.engine.begin() as conn:
            self._lock_resource(conn, resource_id)
            recipe = self._recipe(conn, resource_id)
            previous = conn.execute(text("""
                SELECT *, lease_expires_at > now() AS held FROM catalog_image_generations
                WHERE resource_type = 'recipe' AND resource_id = :id AND stage <> 'retired'
                ORDER BY attempt_number DESC LIMIT 1 FOR UPDATE
            """), {"id": resource_id}).mappings().first()
            if previous and previous["held"]:
                raise ClaimConflict("Redo refused: valid lease")
            successor_id = uuid4()
            if previous:
                # This fenced administrative replacement is atomic with successor creation.
                result = conn.execute(text("""
                    UPDATE catalog_image_generations
                    SET stage = 'retired', superseded_by = :successor, updated_at = now(),
                        lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL
                    WHERE id = :id AND lease_token IS NOT DISTINCT FROM :token
                      AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
                """), {"id": previous["id"], "token": previous["lease_token"],
                       "successor": successor_id})
                if result.rowcount != 1:
                    raise ClaimConflict("Redo lost ownership")
            attempt = self._insert(conn, resource_id, previous["attempt_number"] + 1 if previous else 1,
                                   owner, seconds, note, successor_id)
            return Work(recipe, attempt)

    def update(self, attempt_id: UUID, token: UUID, **changes: Any) -> Attempt:
        changes = clean_changes(changes, self.secrets)
        assignments = []
        params: dict[str, Any] = {"id": attempt_id, "token": token}
        for key, value in changes.items():
            assignments.append(f"{key} = CAST(:{key} AS jsonb)" if key == "settings" else f"{key} = :{key}")
            params[key] = json.dumps(value) if key == "settings" else value
        if changes.get("stage") == Stage.COMPLETED:
            assignments.append("completed_at = now()")
        with self.engine.begin() as conn:
            self._lock_attempt(conn, attempt_id)
            row = conn.execute(text(f"""
                UPDATE catalog_image_generations SET {', '.join(assignments)}, updated_at = now()
                WHERE id = :id AND lease_token = :token AND lease_expires_at > clock_timestamp()
                RETURNING *
            """), params).mappings().first()
            if row is None:
                raise LeaseLost("Lease lost; state write rejected")
            return decode(row)

    def renew(self, attempt_id: UUID, token: UUID, seconds: int) -> None:
        with self.engine.begin() as conn:
            self._lock_attempt(conn, attempt_id)
            result = conn.execute(text("""
                UPDATE catalog_image_generations
                SET lease_expires_at = clock_timestamp() + :seconds * interval '1 second', updated_at = now()
                WHERE id = :id AND lease_token = :token AND lease_expires_at > clock_timestamp()
            """), {"id": attempt_id, "token": token, "seconds": seconds})
            if result.rowcount != 1:
                raise LeaseLost("Lease renewal rejected")

    def release(self, attempt_id: UUID, token: UUID) -> None:
        with self.engine.begin() as conn:
            self._lock_attempt(conn, attempt_id)
            result = conn.execute(text("""
                UPDATE catalog_image_generations
                SET lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL, updated_at = now()
                WHERE id = :id AND lease_token = :token AND lease_expires_at > clock_timestamp()
            """), {"id": attempt_id, "token": token})
            if result.rowcount != 1:
                raise LeaseLost("Lease release rejected")
