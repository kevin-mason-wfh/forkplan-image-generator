import json
from dataclasses import dataclass, fields
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine

from .redaction import redact
from .state import ACTIVE_STAGES, RESUMABLE_AFTER_FAILURE, Attempt, ClaimConflict, LeaseLost, Recipe, Stage


@dataclass(frozen=True)
class Work:
    """A recipe to work on. `attempt` resumes that attempt; `retry_of` starts a new attempt after a failed one."""
    recipe: Recipe
    attempt: Attempt | None = None
    retry_of: Attempt | None = None


class Repository(Protocol):
    def candidates(self, limit: int, ids: tuple[UUID, ...], include_failed: bool) -> list[Work]: ...
    def claim(self, work: Work, owner: str, seconds: int) -> Attempt: ...
    def redo(self, resource_id: UUID, owner: str, seconds: int, note: str | None) -> Work: ...
    def update(self, attempt_id: UUID, token: UUID, **changes: Any) -> Attempt: ...
    def renew(self, attempt_id: UUID, token: UUID, seconds: int) -> None: ...
    def release(self, attempt_id: UUID, token: UUID) -> None: ...
    def release_uncertain(self, resource_id: UUID) -> None: ...


MUTABLE_ATTEMPT_FIELDS = {
    "stage", "last_successful_stage", "description", "description_version", "final_prompt", "prompt_version",
    "settings", "text_usage", "image_usage", "jpeg_sha256", "jpeg_length", "image_key",
    "error_type", "error_message",
}
ATTEMPT_FIELDS = {f.name for f in fields(Attempt)}


def sanitize_changes(changes: dict[str, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    if not changes or not changes.keys() <= MUTABLE_ATTEMPT_FIELDS:
        raise ValueError("Unsupported attempt update")
    return {
        key: redact(str(value), secrets) if key.startswith("error_") and value else value
        for key, value in changes.items()
    }


def attempt_from_row(row: Any) -> Attempt:
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

    def candidates(self, limit: int, ids: tuple[UUID, ...] = (), include_failed: bool = False) -> list[Work]:
        stages = [s.value for s in ACTIVE_STAGES]
        # A failed attempt is only a candidate while it is the recipe's latest attempt.
        failed_filter = ("OR (stage = 'failed' AND NOT EXISTS (SELECT 1 FROM catalog_image_generations newer "
                         "WHERE newer.resource_type = g.resource_type AND newer.resource_id = g.resource_id "
                         "AND newer.attempt_number > g.attempt_number))") if include_failed else ""
        # The id filter is only added when ids are given: an empty IN list has no type and Postgres rejects uuid = integer.
        attempt_filter, recipe_filter = ("AND resource_id IN :ids", "AND r.id IN :ids") if ids else ("", "")
        id_binding = [bindparam("ids", expanding=True)] if ids else []
        with self.engine.connect() as conn:
            attempts = conn.execute(text(f"""
                SELECT * FROM catalog_image_generations g
                WHERE resource_type = 'recipe' AND (stage IN :stages {failed_filter})
                  AND (lease_expires_at IS NULL OR lease_expires_at <= now())
                  {attempt_filter}
                ORDER BY created_at, id LIMIT :limit
            """).bindparams(bindparam("stages", expanding=True), *id_binding),
                {"stages": stages, "limit": limit, **({"ids": ids} if ids else {})}).mappings()
            work = []
            for row in attempts:
                attempt = attempt_from_row(row)
                recipe = self._recipe(conn, row["resource_id"])
                if attempt.stage == Stage.FAILED and attempt.last_successful_stage not in RESUMABLE_AFTER_FAILURE:
                    work.append(Work(recipe, retry_of=attempt))  # needs a new image call: a new attempt, not a resume
                else:
                    work.append(Work(recipe, attempt))
            remaining = limit - len(work)
            if remaining:
                rows = conn.execute(text(f"""
                    SELECT r.id FROM catalog_recipes r
                    WHERE (r.image_key IS NULL OR r.image_key = '')
                      AND r.status IN ('PUBLISHED', 'DRAFT')
                      {recipe_filter}
                      AND NOT EXISTS (
                        SELECT 1 FROM catalog_image_generations g
                        WHERE g.resource_type = 'recipe' AND g.resource_id = r.id)
                    ORDER BY r.id LIMIT :limit
                """).bindparams(*id_binding),
                    {"limit": remaining, **({"ids": ids} if ids else {})}).mappings()
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

    def _latest(self, conn: Connection, resource_id: UUID, *, for_update: bool = False) -> Any:
        return conn.execute(text(f"""
            SELECT *, lease_expires_at > clock_timestamp() AS held FROM catalog_image_generations
            WHERE resource_type = 'recipe' AND resource_id = :id
            ORDER BY attempt_number DESC LIMIT 1 {'FOR UPDATE' if for_update else ''}
        """), {"id": resource_id}).mappings().first()

    def _insert(self, conn: Connection, resource_id: UUID, number: int, owner: str,
                seconds: int, note: str | None = None, retry_of: Attempt | None = None) -> Attempt:
        """A new attempt. A retry carries the earlier description for reuse; it has no text_usage of its own."""
        row = conn.execute(text("""
            INSERT INTO catalog_image_generations
                (id, resource_id, resource_type, attempt_number, stage, last_successful_stage,
                 lease_owner, lease_token, lease_expires_at, redo_note,
                 retry_of_attempt_id, description, description_version)
            VALUES (:id, :resource_id, 'recipe', :number, 'selected', 'selected',
                    :owner, :token, clock_timestamp() + :seconds * interval '1 second', :note,
                    :retry_of, :description, :description_version)
            RETURNING *
        """), {"id": uuid4(), "resource_id": resource_id, "number": number,
               "owner": owner, "token": uuid4(), "seconds": seconds,
               "note": redact(note, self.secrets) if note else None,
               "retry_of": retry_of.id if retry_of else None,
               "description": retry_of.description if retry_of else None,
               "description_version": retry_of.description_version if retry_of else None}).mappings().one()
        return attempt_from_row(row)

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
                      AND (stage <> 'failed' OR last_successful_stage IN ('generated','processed','presigned','uploaded'))
                    RETURNING *
                """), {"owner": owner, "token": uuid4(), "seconds": seconds,
                       "id": work.attempt.id, "expected": work.attempt.stage.value}).mappings().first()
                if row is None:
                    raise ClaimConflict("Attempt already claimed or changed")
                return attempt_from_row(row)
            latest = self._latest(conn, work.recipe.id)
            recipe = self._recipe(conn, work.recipe.id)
            # A new attempt needs no attempt at all, or (for a retry) the failed one it retries as the latest.
            expected = work.retry_of.id if work.retry_of else None
            if (latest["id"] if latest else None) != expected or (latest and latest["stage"] != "failed") \
                    or recipe.image_key or recipe.status not in ("PUBLISHED", "DRAFT"):
                raise ClaimConflict("Recipe is no longer fresh")
            return self._insert(conn, recipe.id, latest["attempt_number"] + 1 if latest else 1,
                                owner, seconds, retry_of=work.retry_of)

    def redo(self, resource_id: UUID, owner: str, seconds: int, note: str | None) -> Work:
        with self.engine.begin() as conn:
            self._lock_resource(conn, resource_id)
            recipe = self._recipe(conn, resource_id)
            previous = self._latest(conn, resource_id, for_update=True)
            if previous and previous["held"]:
                raise ClaimConflict("Redo refused: valid lease")
            if previous and previous["stage"] == Stage.UNCERTAIN:
                raise ClaimConflict("Redo refused: the earlier image request may have been billed; "
                                    "check the provider, then run --release-uncertain")
            if previous and previous["stage"] in {s.value for s in ACTIVE_STAGES}:
                # An unfinished attempt with an expired lease is set aside (completed and failed rows stay as they are).
                result = conn.execute(text("""
                    UPDATE catalog_image_generations
                    SET stage = 'abandoned', updated_at = now(),
                        lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL
                    WHERE id = :id AND lease_token IS NOT DISTINCT FROM :token
                      AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
                """), {"id": previous["id"], "token": previous["lease_token"]})
                if result.rowcount != 1:
                    raise ClaimConflict("Redo lost ownership")
            attempt = self._insert(conn, resource_id, previous["attempt_number"] + 1 if previous else 1,
                                   owner, seconds, note, attempt_from_row(previous) if previous else None)
            return Work(recipe, attempt)

    def release_uncertain(self, resource_id: UUID) -> None:
        """After checking the provider, let an uncertain attempt be redone. History is kept (stage 'abandoned')."""
        with self.engine.begin() as conn:
            self._lock_resource(conn, resource_id)
            result = conn.execute(text("""
                UPDATE catalog_image_generations
                SET stage = 'abandoned', updated_at = now(),
                    lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL
                WHERE resource_type = 'recipe' AND resource_id = :id AND stage = 'generation_uncertain'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
            """), {"id": resource_id})
            if result.rowcount != 1:
                raise ClaimConflict("No released-able uncertain attempt (none exists, or its lease is still held)")

    def update(self, attempt_id: UUID, token: UUID, **changes: Any) -> Attempt:
        changes = sanitize_changes(changes, self.secrets)
        assignments = []
        params: dict[str, Any] = {"id": attempt_id, "token": token}
        for key, value in changes.items():
            is_json = key in ("settings", "text_usage", "image_usage")
            assignments.append(f"{key} = CAST(:{key} AS jsonb)" if is_json else f"{key} = :{key}")
            params[key] = json.dumps(value) if is_json else value
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
            return attempt_from_row(row)

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
