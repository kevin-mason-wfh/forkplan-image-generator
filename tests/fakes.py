from copy import deepcopy
from threading import RLock
from time import monotonic
from uuid import uuid4

from forkplan_images.repository import Work, clean_changes
from forkplan_images.redaction import redact
from forkplan_images.state import ACTIVE, Attempt, ClaimConflict, LeaseLost, Stage


class MemoryRepository:
    def __init__(self, recipes, clock=monotonic, secrets=()):
        self.recipes = {r.id: r for r in recipes}
        self.attempts = {}
        self.clock = clock
        self.secrets = secrets
        self.lock = RLock()
        self.writes = []

    def candidates(self, limit, ids=(), retry=False):
        with self.lock:
            attempts = [
                a for a in self.attempts.values()
                if (a.stage in ACTIVE or (retry and a.stage == Stage.FAILED))
                and not self.held(a) and (not ids or a.resource_id in ids)
            ]
            work = [Work(self.recipes[a.resource_id], deepcopy(a)) for a in attempts]
            work += [
                Work(r) for r in self.recipes.values()
                if (not ids or r.id in ids) and not r.image_key
                and r.status in ("PUBLISHED", "DRAFT")
                and not any(a.resource_id == r.id and a.stage != Stage.RETIRED
                            for a in self.attempts.values())
            ]
            return work[:limit]

    def held(self, a):
        return a.lease_expires_at is not None and a.lease_expires_at > self.clock()

    def claim(self, work, owner, seconds):
        with self.lock:
            if work.attempt:
                a = self.attempts[work.attempt.id]
                if self.held(a) or a.stage != work.attempt.stage:
                    raise ClaimConflict("Already claimed")
                if a.stage == Stage.FAILED:
                    a.stage = a.last_successful_stage
                    a.retry_count += 1
                    a.error_type = a.error_message = None
            else:
                if any(a.resource_id == work.recipe.id and a.stage != Stage.RETIRED
                       for a in self.attempts.values()):
                    raise ClaimConflict("Duplicate")
                a = Attempt(work.recipe.id)
                self.attempts[a.id] = a
            a.lease_owner, a.lease_token = owner, uuid4()
            a.lease_expires_at = self.clock() + seconds
            self.writes.append(("claim", a.id))
            return deepcopy(a)

    def redo(self, resource_id, owner, seconds, note):
        with self.lock:
            previous = [a for a in self.attempts.values()
                        if a.resource_id == resource_id and a.stage != Stage.RETIRED]
            old = max(previous, key=lambda a: a.attempt_number) if previous else None
            if old and self.held(old):
                raise ClaimConflict("Valid lease")
            a = Attempt(resource_id, attempt_number=old.attempt_number + 1 if old else 1,
                        redo_note=redact(note, self.secrets) if note else None)
            if old:
                old.stage, old.superseded_by = Stage.RETIRED, a.id
                old.lease_token = old.lease_owner = old.lease_expires_at = None
            self.attempts[a.id] = a
            return Work(self.recipes[resource_id], self.claim(Work(self.recipes[resource_id], a),
                                                           owner, seconds))

    def owned(self, attempt_id, token):
        a = self.attempts[attempt_id]
        if a.lease_token != token or not self.held(a):
            raise LeaseLost("Lease lost")
        return a

    def update(self, attempt_id, token, **changes):
        with self.lock:
            a = self.owned(attempt_id, token)
            changes = clean_changes(changes, self.secrets)
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
