from dataclasses import asdict
from unittest.mock import MagicMock
from uuid import uuid4

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


@pytest.mark.parametrize("stage", [Stage.FAILED, Stage.COMPLETED, Stage.PRESIGNED])
def test_redo_adds_a_linked_attempt_and_keeps_history(stage):
    recipe = Recipe(uuid4(), "Soup", image_key="")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=stage, description="A described dish.", description_version="v1")
    with pytest.raises(ClaimConflict):  # lease still held
        repo.redo(recipe.id, "two", 10, None)
    repo.release(a.id, a.lease_token)
    assert bool(repo.candidates(10)) == (stage == Stage.PRESIGNED)  # unfinished work resumes; the rest is left alone
    next_work = repo.redo(recipe.id, "two", 10, "manual")
    new = next_work.attempt
    assert new.attempt_number == 2 and new.retry_of_attempt_id == a.id
    assert new.description == "A described dish." and new.text_usage is None
    # Only an unfinished attempt is set aside; failed and completed ones keep their outcome.
    assert repo.attempts[a.id].stage == (Stage.ABANDONED if stage == Stage.PRESIGNED else stage)


def test_uncertain_attempt_reserves_the_recipe_until_released():
    recipe = Recipe(uuid4(), "Soup", image_key="")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=Stage.UNCERTAIN)
    repo.release(a.id, a.lease_token)
    assert not repo.candidates(10, include_failed=True)
    with pytest.raises(ClaimConflict):
        repo.redo(recipe.id, "two", 10, None)
    repo.release_uncertain(recipe.id)
    assert repo.attempts[a.id].stage == Stage.ABANDONED
    assert repo.redo(recipe.id, "two", 10, None).attempt.attempt_number == 2


def test_failure_before_the_image_is_saved_starts_a_new_attempt():
    recipe = Recipe(uuid4(), "Soup")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=Stage.FAILED, last_successful_stage=Stage.PROMPT_BUILT,
                description="A described dish.", description_version="v1")
    repo.release(a.id, a.lease_token)
    (work,) = repo.candidates(10, include_failed=True)
    assert work.attempt is None and work.retry_of.id == a.id  # not resumed in place
    new = repo.claim(work, "two", 10)
    assert new.retry_of_attempt_id == a.id and new.attempt_number == 2 and new.description == "A described dish."
    assert repo.attempts[a.id].stage == Stage.FAILED and repo.attempts[a.id].retry_count == 0


def test_retry_keeps_progress_and_clears_errors():
    recipe = Recipe(uuid4(), "Soup")
    repo = MemoryRepository([recipe])
    a = repo.claim(Work(recipe), "one", 10)
    repo.update(a.id, a.lease_token, stage=Stage.FAILED,
                last_successful_stage=Stage.PROCESSED, error_message="failure")
    repo.release(a.id, a.lease_token)
    retried = repo.claim(repo.candidates(1, include_failed=True)[0], "two", 10)
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


@pytest.mark.parametrize("ids", [(), (uuid4(),)])
def test_candidate_queries_only_filter_by_id_when_ids_are_given(ids):
    """An empty IN list is untyped and fails on Postgres (uuid = integer); found by running against a real database."""
    statements = []
    conn = MagicMock()
    conn.execute.side_effect = lambda statement, params=None: (
        statements.append(str(statement)), MagicMock(mappings=lambda: []))[1]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    PostgresRepository(engine).candidates(5, ids)
    assert len(statements) == 2 and all(("POSTCOMPILE_ids" in s) == bool(ids) for s in statements)
    assert all("all_ids" not in s for s in statements)
