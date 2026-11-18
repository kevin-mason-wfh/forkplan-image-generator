from copy import deepcopy
from threading import RLock
from time import monotonic
from uuid import uuid4

from forkplan_images.redaction import redact
from forkplan_images.repository import Work, sanitize_changes
from forkplan_images.state import (
    ACTIVE_STAGES,
    MAX_ATTEMPTS,
    RESUMABLE_AFTER_FAILURE,
    Attempt,
    ClaimConflict,
    LeaseLost,
    Stage,
)


class MemoryRepository:
    def __init__(self, recipes, clock=monotonic, secrets=()):
        self.recipes = {r.id: r for r in recipes}
        self.attempts = {}
        self.clock = clock
        self.secrets = secrets
        self.lock = RLock()
        self.writes = []

    def candidates(self, limit, ids=(), include_failed=False):
        with self.lock:
            work = []
            for a in self.attempts.values():
                if self.held(a) or (ids and a.resource_id not in ids):
                    continue
                recipe = self.recipes[a.resource_id]
                if a.stage in ACTIVE_STAGES:
                    work.append(Work(recipe, deepcopy(a)))
                elif (include_failed and a.stage == Stage.FAILED and self.latest(a.resource_id) is a
                      and a.attempt_number < MAX_ATTEMPTS and a.retry_count < MAX_ATTEMPTS):
                    if a.last_successful_stage in RESUMABLE_AFTER_FAILURE:
                        work.append(Work(recipe, deepcopy(a)))
                    else:
                        work.append(Work(recipe, retry_of=deepcopy(a)))
            work += [
                Work(r) for r in self.recipes.values()
                if (not ids or r.id in ids) and not r.image_key
                and r.status in ("PUBLISHED", "DRAFT") and self.latest(r.id) is None
            ]
            return work[:limit]

    def latest(self, resource_id):
        found = [a for a in self.attempts.values() if a.resource_id == resource_id]
        return max(found, key=lambda a: a.attempt_number) if found else None

    def held(self, a):
        return a.lease_expires_at is not None and a.lease_expires_at > self.clock()

    def _new_attempt(self, resource_id, retry_of=None, note=None):
        latest = self.latest(resource_id)
        a = Attempt(resource_id, attempt_number=latest.attempt_number + 1 if latest else 1,
                    retry_of_attempt_id=retry_of.id if retry_of else None,
                    description=retry_of.description if retry_of else None,
                    description_version=retry_of.description_version if retry_of else None,
                    redo_note=redact(note, self.secrets) if note else None)
        self.attempts[a.id] = a
        return a

    def claim(self, work, owner, seconds):
        with self.lock:
            if work.attempt:
                a = self.attempts[work.attempt.id]
                if self.held(a) or a.stage != work.attempt.stage:
                    raise ClaimConflict("Already claimed")
                if a.stage == Stage.FAILED:
                    if a.last_successful_stage not in RESUMABLE_AFTER_FAILURE:
                        raise ClaimConflict("Failed before the image was saved")
                    a.stage = a.last_successful_stage
                    a.retry_count += 1
                    a.error_type = a.error_message = None
            else:
                latest = self.latest(work.recipe.id)
                expected = work.retry_of.id if work.retry_of else None
                if (latest.id if latest else None) != expected or (latest and latest.stage != Stage.FAILED):
                    raise ClaimConflict("Duplicate")
                a = self._new_attempt(work.recipe.id, work.retry_of)
            a.lease_owner, a.lease_token = owner, uuid4()
            a.lease_expires_at = self.clock() + seconds
            self.writes.append(("claim", a.id))
            return deepcopy(a)

    def redo(self, resource_id, owner, seconds, note):
        with self.lock:
            old = self.latest(resource_id)
            if old and self.held(old):
                raise ClaimConflict("Valid lease")
            if old and old.stage == Stage.UNCERTAIN:
                raise ClaimConflict("Redo refused: uncertain")
            if old and old.stage in ACTIVE_STAGES:
                old.stage = Stage.ABANDONED
                old.lease_token = old.lease_owner = old.lease_expires_at = None
            a = self._new_attempt(resource_id, old, note)
            return Work(self.recipes[resource_id], self.claim(Work(self.recipes[resource_id], a),
                                                           owner, seconds))

    def release_uncertain(self, resource_id):
        with self.lock:
            old = self.latest(resource_id)
            if not old or old.stage != Stage.UNCERTAIN or self.held(old):
                raise ClaimConflict("No uncertain attempt to release")
            old.stage = Stage.ABANDONED
            old.lease_token = old.lease_owner = old.lease_expires_at = None

    def owned(self, attempt_id, token):
        a = self.attempts[attempt_id]
        if a.lease_token != token or not self.held(a):
            raise LeaseLost("Lease lost")
        return a

    def update(self, attempt_id, token, **changes):
        with self.lock:
            a = self.owned(attempt_id, token)
            changes = sanitize_changes(changes, self.secrets)
            for key, value in changes.items():
                setattr(a, key, value)
            self.writes.append(("update", deepcopy(changes)))
            return deepcopy(a)

    def renew(self, attempt_id, token, seconds):
        with self.lock:
            self.owned(attempt_id, token).lease_expires_at = self.clock() + seconds
            self.writes.append(("renew", attempt_id))

    def release(self, attempt_id, token):
        with self.lock:
            a = self.owned(attempt_id, token)
            a.lease_owner = a.lease_token = a.lease_expires_at = None
            self.writes.append(("release", attempt_id))


def plain(text: str) -> str:
    """A clean description padded past the minimum length (no banned words), for tests that fake the text model."""
    return text + " The surface is matte and slightly uneven." * 6
