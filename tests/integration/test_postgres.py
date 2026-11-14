import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fakes import plain
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from test_storage_imaging import png

from forkplan_images.orchestrator import Options, Orchestrator
from forkplan_images.pricing import build_usage
from forkplan_images.repository import PostgresRepository
from forkplan_images.state import ClaimConflict, LeaseLost, Stage
from forkplan_images.storage import LocalStorage
from forkplan_images.uploads import Presign


@pytest.fixture
def repo(engine):
    return PostgresRepository(engine, ())


def test_candidates_with_and_without_ids(repo, recipe_ids):
    assert len(repo.candidates(10)) == 3  # an empty id list must not break the query
    assert [w.recipe.id for w in repo.candidates(10, (recipe_ids[1],))] == [recipe_ids[1]]


def test_claim_lease_expiry_retry_and_redo(repo, recipe_ids):
    work = repo.candidates(1)[0]
    attempt = repo.claim(work, "one", 1)
    with pytest.raises(ClaimConflict):
        repo.claim(work, "two", 1)
    attempt = repo.update(attempt.id, attempt.lease_token, stage=Stage.DESCRIBED, last_successful_stage=Stage.DESCRIBED,
                          description="d" * 30, description_version="v", text_usage={"model": "m", "input_tokens": 1})
    assert attempt.text_usage == {"model": "m", "input_tokens": 1} and attempt.description_version == "v"
    time.sleep(1.2)
    with pytest.raises(LeaseLost):  # the lease expired: even the owner's write is rejected
        repo.update(attempt.id, attempt.lease_token, stage=Stage.PROMPT_BUILT)
    resumed = repo.claim(next(w for w in repo.candidates(10) if w.attempt), "two", 60)
    assert resumed.lease_token != attempt.lease_token and resumed.stage == Stage.DESCRIBED
    failed = repo.update(resumed.id, resumed.lease_token, stage=Stage.FAILED, last_successful_stage=Stage.GENERATED,
                         error_type="E", error_message="boom")
    repo.release(failed.id, failed.lease_token)
    assert not [w for w in repo.candidates(10) if w.attempt]
    reopened = repo.claim(next(w for w in repo.candidates(10, (), True) if w.attempt), "three", 60)
    assert reopened.stage == Stage.GENERATED and reopened.retry_count == 1 and reopened.error_message is None
    repo.release(reopened.id, reopened.lease_token)
    redo = repo.redo(work.recipe.id, "admin", 60, "reviewed")
    assert redo.attempt.attempt_number == 2 and redo.attempt.redo_note == "reviewed"
    assert redo.attempt.retry_of_attempt_id == reopened.id and redo.attempt.description == "d" * 30
    assert redo.attempt.text_usage is None  # no text call ran for the redo
    with pytest.raises(ClaimConflict):
        repo.redo(work.recipe.id, "admin2", 60, None)


def test_failure_before_the_image_is_saved_becomes_a_new_attempt(repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "one", 60)
    failed = repo.update(attempt.id, attempt.lease_token, stage=Stage.FAILED, last_successful_stage=Stage.PROMPT_BUILT,
                         description="d" * 30, description_version="v")
    repo.release(failed.id, failed.lease_token)
    (work,) = repo.candidates(10, (attempt.resource_id,), True)
    assert work.attempt is None and work.retry_of.id == failed.id
    retry = repo.claim(work, "two", 60)
    assert retry.attempt_number == 2 and retry.retry_of_attempt_id == failed.id and retry.description == "d" * 30
    assert not [w for w in repo.candidates(10, (attempt.resource_id,), True) if w.retry_of]  # the failed row is no longer latest


def test_uncertain_attempt_blocks_redo_until_released(repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "one", 60)
    uncertain = repo.update(attempt.id, attempt.lease_token, stage=Stage.UNCERTAIN)
    repo.release(uncertain.id, uncertain.lease_token)
    with pytest.raises(ClaimConflict):
        repo.redo(attempt.resource_id, "two", 60, None)
    repo.release_uncertain(attempt.resource_id)
    assert repo.redo(attempt.resource_id, "two", 60, None).attempt.retry_of_attempt_id == attempt.id


def test_redo_sets_aside_only_unfinished_attempts(engine, repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "one", 1)
    time.sleep(1.2)
    redo = repo.redo(attempt.resource_id, "two", 60, None)
    with engine.connect() as conn:
        stage = conn.execute(text("SELECT stage FROM catalog_image_generations WHERE id = :id"),
                             {"id": attempt.id}).scalar_one()
    assert stage == "abandoned" and redo.attempt.retry_of_attempt_id == attempt.id


def test_database_rejects_inconsistent_rows(engine, repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "one", 60)
    for statement in ("UPDATE catalog_image_generations SET lease_owner = NULL",                 # token/expiry still set
                      "UPDATE catalog_image_generations SET text_usage = '[1]'::jsonb",           # not an object
                      "UPDATE catalog_image_generations SET image_usage = '{\"input_tokens\": \"many\"}'::jsonb",  # view would fail
                      "UPDATE catalog_image_generations SET image_usage = '{\"cost_usd\": \"free\"}'::jsonb",
                      "UPDATE catalog_image_generations SET stage = 'completed'",                 # completed needs image_key
                      "UPDATE catalog_image_generations SET jpeg_length = 5000000"):              # over the JPEG limit
        with pytest.raises(DBAPIError), engine.begin() as conn:
            conn.execute(text(statement))
    assert attempt.id


def test_orchestrator_records_tokens_and_cost_in_the_database(engine, repo, recipe_ids, tmp_path: Path):
    describe, generate, upload = Mock(), Mock(), Mock()
    describe.describe.return_value = plain("A plain plate holding the cooked dish.")
    describe.last_usage = build_usage("openai", "gpt-5.4-mini", input_tokens=900, output_text_tokens=300)
    generate.generate.return_value = png()
    generate.last_usage = build_usage("openai", "gpt-image-2.6-flare", input_tokens=560, output_image_tokens=1449)
    upload.presign.side_effect = lambda rid, name, size, checkpoint=None: Presign(
        upload_url="https://s3.example.test/x", public_url="https://c.example.test/x",
        image_key=f"recipes/{rid}/{name}", content_type="image/jpeg", expires_in=300)
    runner = Orchestrator(repo, LocalStorage(tmp_path), describe, generate, upload, {"text_model": "m"}, 300)
    summary = runner.run(Options(limit=3, max_images=1))
    assert [r.status for r in summary.results] == ["completed", "skipped", "skipped"] and summary.total_tokens == 3209
    with engine.connect() as conn:
        row = conn.execute(text("SELECT stage, total_tokens, total_cost_usd, description_version, unpriced_calls "
                                "FROM catalog_image_generation_costs")).one()
    assert row.stage == "completed" and row.total_tokens == 3209 and float(row.total_cost_usd) == pytest.approx(0.048295)
    assert row.description_version and row.unpriced_calls == 0

    describe.describe.side_effect = ValueError("Description still has style problems: glossy")
    runner.run(Options(limit=3, max_images=1))  # a rejected description still saves the billed text call
    with engine.connect() as conn:
        failed = conn.execute(text("SELECT total_tokens FROM catalog_image_generation_costs WHERE stage = 'failed'")).all()
    assert [row.total_tokens for row in failed] == [1200, 1200]  # the two remaining recipes both failed after one billed call
