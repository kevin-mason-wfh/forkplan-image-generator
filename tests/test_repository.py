from uuid import uuid4
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from fakes import MemoryRepository
from forkplan_images.repository import PostgresRepository, Work
from forkplan_images.state import Attempt, ClaimConflict, LeaseLost, Recipe, Stage


def test_claim_expiry_and_token_fencing():
    now = [0]
    recipe = Recipe(uuid4(), "Soup")
    repo = MemoryRepository([recipe], clock=lambda: now[0])
    work = Work(recipe)
    first = repo.claim(work, "one", 10)
    with pytest.raises(ClaimConflict):
        repo.claim(work, "two", 10)
    now[0] = 11
    second = repo.claim(repo.candidates(1)[0], "two", 10)
    assert first.lease_token != second.lease_token
    with pytest.raises(LeaseLost):
        repo.update(first.id, first.lease_token, stage=Stage.DESCRIBED)
    repo.renew(second.id, second.lease_token, 20)
    assert repo.attempts[first.id].lease_expires_at == 31


@pytest.mark.parametrize("stage", [Stage.FAILED, Stage.UNCERTAIN, Stage.COMPLETED])
def test_terminal_exclusion_retry_and_redo(stage):
    recipe = Recipe(uuid4(), "Soup", image_key="")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=stage)
    with pytest.raises(ClaimConflict):
        repo.redo(recipe.id, "two", 10, None)
    repo.release(a.id, a.lease_token)
    assert not repo.candidates(10)
    assert bool(repo.candidates(10, retry=True)) == (stage == Stage.FAILED)
    next_work = repo.redo(recipe.id, "two", 10, "manual")
    assert repo.attempts[a.id].superseded_by == next_work.attempt.id
    assert repo.attempts[a.id].stage == Stage.RETIRED
    assert next_work.attempt.attempt_number == 2


def test_retry_keeps_progress_and_clears_errors():
    recipe = Recipe(uuid4(), "Soup")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=Stage.FAILED,
                last_successful_stage=Stage.PROCESSED, error_message="failure")
    repo.release(a.id, a.lease_token)
    retried = repo.claim(repo.candidates(1, retry=True)[0], "two", 10)
    assert retried.stage == Stage.PROCESSED
    assert retried.retry_count == 1
    assert retried.error_message is None


def test_postgres_update_is_parameterized_fenced_and_redacted():
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    attempt = Attempt(uuid4())
    connection.execute.return_value.mappings.return_value.first.return_value = asdict(attempt)
    repository = PostgresRepository(engine, ("private-placeholder",))
    message = "private-placeholder https://bucket.example.test/?signature=private-placeholder"
    repository.update(attempt.id, uuid4(), stage=Stage.FAILED, error_message=message)
    calls = connection.execute.call_args_list
    assert "FOR UPDATE" in str(calls[0].args[0])
    sql, params = calls[1].args
    assert "lease_token = :token" in str(sql)
    assert "lease_expires_at > clock_timestamp()" in str(sql)
    assert message not in str(sql)
    assert "private-placeholder" not in params["error_message"]
    assert "https://" not in params["error_message"]
    connection.execute.return_value.mappings.return_value.first.return_value = None
    with pytest.raises(LeaseLost):
        repository.update(attempt.id, uuid4(), stage=Stage.COMPLETED)


@pytest.mark.parametrize("method", ["renew", "release"])
def test_postgres_lease_mutation_fenced_after_lock(method):
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.rowcount = 0
    repository = PostgresRepository(engine)
    with pytest.raises(LeaseLost):
        args = (uuid4(), uuid4(), 300) if method == "renew" else (uuid4(), uuid4())
        getattr(repository, method)(*args)
    calls = connection.execute.call_args_list
    assert "FOR UPDATE" in str(calls[0].args[0])
    assert "lease_token = :token" in str(calls[1].args[0])
    assert "clock_timestamp()" in str(calls[1].args[0])
