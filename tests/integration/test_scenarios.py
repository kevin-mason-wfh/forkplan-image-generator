"""Scenarios and edge cases against a real Postgres: constraints, queue rules, races, redo/retry, cost view."""
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from fakes import plain
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from test_storage_imaging import png

from forkplan_images.generation import GenerationError
from forkplan_images.orchestrator import Options, Orchestrator
from forkplan_images.pricing import build_usage
from forkplan_images.repository import PostgresRepository
from forkplan_images.state import ClaimConflict, LeaseLost, Stage
from forkplan_images.storage import LocalStorage
from forkplan_images.uploads import Presign


@pytest.fixture
def repo(engine):
    return PostgresRepository(engine, ())


def insert_attempt(engine, recipe_id, number=1, stage="selected", **extra):
    columns = {"id": uuid.uuid4(), "resource_id": recipe_id, "resource_type": "recipe", "attempt_number": number,
               "stage": stage, "last_successful_stage": "selected", **extra}
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO catalog_image_generations ({', '.join(columns)}) "
                          f"VALUES ({', '.join(':' + c for c in columns)})"), columns)
    return columns["id"]


def stage_of(engine, attempt_id):
    with engine.connect() as conn:
        return conn.execute(text("SELECT stage FROM catalog_image_generations WHERE id = :id"),
                            {"id": attempt_id}).scalar_one()


# -- constraints ---------------------------------------------------------------------------------------------

def test_only_one_in_flight_attempt_per_recipe(engine, recipe_ids):
    first = insert_attempt(engine, recipe_ids[0])
    with pytest.raises(DBAPIError):
        insert_attempt(engine, recipe_ids[0], number=2)  # a second in-flight attempt
    insert_attempt(engine, recipe_ids[1])  # another recipe is independent
    with engine.begin() as conn:  # an uncertain attempt still reserves the recipe...
        conn.execute(text("UPDATE catalog_image_generations SET stage = 'generation_uncertain' WHERE id = :id"), {"id": first})
    with pytest.raises(DBAPIError):
        insert_attempt(engine, recipe_ids[0], number=2)
    with engine.begin() as conn:  # ...until it is set aside; failed and completed rows never block
        conn.execute(text("UPDATE catalog_image_generations SET stage = 'abandoned' WHERE id = :id"), {"id": first})
    second = insert_attempt(engine, recipe_ids[0], number=2, retry_of_attempt_id=first)
    with engine.begin() as conn:
        conn.execute(text("UPDATE catalog_image_generations SET stage = 'failed' WHERE id = :id"), {"id": second})
    insert_attempt(engine, recipe_ids[0], number=3, retry_of_attempt_id=second)


def test_attempt_numbers_links_and_enums_are_enforced(engine, recipe_ids):
    first = insert_attempt(engine, recipe_ids[0], stage="failed")
    for bad in ({"number": 1},                                                  # duplicate attempt number
                {"number": 0, "stage": "failed"},                               # attempt numbers start at 1
                {"number": 2, "stage": "failed", "retry_of_attempt_id": uuid.uuid4()},  # link to a missing attempt
                {"number": 2, "stage": "retired"},                              # removed stage
                {"number": 2, "stage": "failed", "resource_type": "menu"}):
        with pytest.raises(DBAPIError):
            insert_attempt(engine, recipe_ids[0], **bad)
    assert first


@pytest.mark.parametrize("usage", ['{"input_tokens": "many"}', '{"input_tokens": 1.5}', '{"output_text_tokens": -1}',
                                   '{"cost_usd": "free"}', '{"reasoning_tokens": null, "cost_usd": [1]}', "[1]", '"x"'])
def test_invalid_usage_is_rejected(engine, recipe_ids, usage):
    with pytest.raises(DBAPIError):
        insert_attempt(engine, recipe_ids[0], text_usage=usage)


def test_valid_usage_shapes_are_accepted(engine, recipe_ids):
    insert_attempt(engine, recipe_ids[0], text_usage='{"input_tokens": 5, "cost_usd": null, "rates": {"in": 0.75}}',
                   image_usage=None)
    insert_attempt(engine, recipe_ids[1], text_usage='{"cost_usd": 1.2e-05, "model": "m"}', image_usage="{}")


# -- cost view -----------------------------------------------------------------------------------------------

def test_view_separates_skipped_calls_from_unpriced_ones(engine, recipe_ids):
    priced = build_usage("openai", "gpt-5.4-mini", input_tokens=1000, output_text_tokens=100)
    unpriced = build_usage("openai", "model-with-no-rate", input_tokens=50, output_image_tokens=7)
    assert unpriced["cost_usd"] is None
    import json
    ids = {
        "skipped_text": insert_attempt(engine, recipe_ids[0], stage="failed", image_usage=json.dumps(priced)),
        "unpriced_image": insert_attempt(engine, recipe_ids[1], stage="failed", text_usage=json.dumps(priced),
                                         image_usage=json.dumps(unpriced)),
        "nothing": insert_attempt(engine, recipe_ids[2], stage="failed"),
    }
    with engine.connect() as conn:
        rows = {r.attempt_id: r for r in conn.execute(text("SELECT * FROM catalog_image_generation_costs"))}
    assert rows[ids["skipped_text"]].unpriced_calls == 0 and rows[ids["skipped_text"]].text_model is None
    assert rows[ids["unpriced_image"]].unpriced_calls == 1
    assert rows[ids["unpriced_image"]].total_tokens == 1100 + 57
    assert float(rows[ids["unpriced_image"]].total_cost_usd) == pytest.approx(float(priced["cost_usd"]))
    assert rows[ids["nothing"]].total_tokens == 0 and rows[ids["nothing"]].unpriced_calls == 0


# -- selection rules -----------------------------------------------------------------------------------------

def test_fresh_selection_rules(engine, repo, recipe_ids):
    with engine.begin() as conn:
        conn.execute(text("UPDATE catalog_recipes SET image_key = 'has/image.jpg' WHERE id = :id"), {"id": recipe_ids[0]})
        conn.execute(text("UPDATE catalog_recipes SET status = 'RETIRED' WHERE id = :id"), {"id": recipe_ids[1]})
        conn.execute(text("UPDATE catalog_recipes SET image_key = '' WHERE id = :id"), {"id": recipe_ids[2]})
    assert [w.recipe.id for w in repo.candidates(10)] == [recipe_ids[2]]  # an empty key counts as missing
    attempt = repo.claim(repo.candidates(1)[0], "w", 60)
    repo.update(attempt.id, attempt.lease_token, stage=Stage.ABANDONED)
    repo.release(attempt.id, attempt.lease_token)
    assert repo.candidates(10) == []  # any earlier attempt (even abandoned) means: not fresh, use --redo
    assert repo.candidates(10, (recipe_ids[0],)) == []
    assert repo.candidates(0) == []


def test_two_workers_cannot_claim_the_same_recipe(repo, recipe_ids):
    work = repo.candidates(1)[0]

    def attempt_claim(_):
        try:
            return repo.claim(work, "w", 60)
        except ClaimConflict:
            return None

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(attempt_claim, range(8)))
    assert len([r for r in results if r]) == 1


def test_expired_lease_fences_every_write_and_a_new_worker_can_resume(repo, recipe_ids):
    import time
    first = repo.claim(repo.candidates(1)[0], "one", 1)
    time.sleep(1.2)
    for call in (lambda: repo.renew(first.id, first.lease_token, 5), lambda: repo.release(first.id, first.lease_token),
                 lambda: repo.update(first.id, first.lease_token, stage=Stage.DESCRIBED)):
        with pytest.raises(LeaseLost):
            call()
    (work,) = [w for w in repo.candidates(10, (first.resource_id,)) if w.attempt]
    second = repo.claim(work, "two", 60)
    assert second.id == first.id and second.lease_token != first.lease_token
    with pytest.raises(LeaseLost):
        repo.update(first.id, first.lease_token, stage=Stage.DESCRIBED)  # the old owner is fenced out


def test_release_uncertain_requires_an_uncertain_attempt(repo, recipe_ids):
    with pytest.raises(ClaimConflict):
        repo.release_uncertain(recipe_ids[0])  # no attempt
    attempt = repo.claim(repo.candidates(1)[0], "w", 60)
    repo.release(attempt.id, attempt.lease_token)
    with pytest.raises(ClaimConflict):
        repo.release_uncertain(attempt.resource_id)  # in flight, not uncertain


# -- redo ----------------------------------------------------------------------------------------------------

def test_redo_keeps_a_completed_attempt_completed_and_links_the_new_one(engine, repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "w", 60)
    done = repo.update(attempt.id, attempt.lease_token, stage=Stage.COMPLETED, last_successful_stage=Stage.COMPLETED,
                       image_key="recipes/x.jpg", description="d" * 30, description_version="v1",
                       text_usage={"model": "m", "input_tokens": 3, "cost_usd": 0.1})
    repo.release(done.id, done.lease_token)
    redo = repo.redo(done.resource_id, "admin", 60, None)
    assert stage_of(engine, done.id) == "completed"
    assert redo.attempt.retry_of_attempt_id == done.id and redo.attempt.attempt_number == 2
    assert redo.attempt.description == "d" * 30 and redo.attempt.text_usage is None and redo.attempt.image_key is None
    assert not [w for w in repo.candidates(10, (done.resource_id,)) if w.attempt and w.attempt.id == done.id]


def test_redo_for_a_recipe_with_no_history_and_an_existing_image(engine, repo, recipe_ids):
    with engine.begin() as conn:
        conn.execute(text("UPDATE catalog_recipes SET image_key = 'old.jpg' WHERE id = :id"), {"id": recipe_ids[0]})
    work = repo.redo(recipe_ids[0], "admin", 60, "manual")
    assert work.attempt.attempt_number == 1 and work.attempt.retry_of_attempt_id is None and work.attempt.description is None


def test_redo_over_an_unfinished_attempt_with_an_expired_lease_sets_it_aside(engine, repo, recipe_ids):
    import time
    first = repo.claim(repo.candidates(1)[0], "w", 1)
    time.sleep(1.2)
    redo = repo.redo(first.resource_id, "admin", 60, None)
    assert stage_of(engine, first.id) == "abandoned" and redo.attempt.retry_of_attempt_id == first.id


def test_redo_is_refused_while_another_worker_holds_the_lease(repo, recipe_ids):
    attempt = repo.claim(repo.candidates(1)[0], "w", 60)
    with pytest.raises(ClaimConflict):
        repo.redo(attempt.resource_id, "admin", 60, None)


# -- full pipeline through the orchestrator ------------------------------------------------------------------

def pipeline(repo, tmp_path: Path):
    describe, generate, upload = Mock(), Mock(), Mock()
    describe.describe.return_value = plain("A plain plate holding the cooked dish.")
    describe.last_usage = build_usage("openai", "gpt-5.4-mini", input_tokens=900, output_text_tokens=300)
    generate.generate.return_value = png()
    generate.last_usage = build_usage("openai", "gpt-image-2.6-flare", input_tokens=560, output_image_tokens=1449)
    counter = iter(range(1, 100))
    upload.presign.side_effect = lambda rid, name, size, checkpoint=None: Presign(
        upload_url="https://s3.example.test/x", public_url="https://c.example.test/x",
        image_key=f"recipes/{rid}/{name}-{next(counter)}", content_type="image/jpeg", expires_in=300)
    return Orchestrator(repo, LocalStorage(tmp_path), describe, generate, upload, {"text_model": "m"}, 300), \
        describe, generate, upload


def rows(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM catalog_image_generations ORDER BY created_at, attempt_number")).all()


def test_upload_failure_resumes_the_same_attempt_without_a_new_image(engine, repo, recipe_ids, tmp_path):
    runner, describe, generate, upload = pipeline(repo, tmp_path)
    upload.upload.side_effect = RuntimeError("PUT failed")
    first = runner.run(Options(limit=1, max_images=1))
    assert first.exit_code == 1
    (failed,) = rows(engine)
    assert failed.stage == "failed" and failed.last_successful_stage == "presigned" and failed.error_type == "RuntimeError"
    upload.upload.side_effect = None
    second = runner.run(Options(limit=1, max_images=0, retry_failures=True))
    assert second.exit_code == 0
    (done,) = rows(engine)
    assert done.id == failed.id and done.stage == "completed" and done.retry_count == 1 and done.error_message is None
    assert generate.generate.call_count == 1 and done.image_key and done.completed_at


def test_failure_before_the_image_is_saved_creates_a_linked_attempt_that_reuses_the_description(
        engine, repo, recipe_ids, tmp_path):
    runner, describe, generate, upload = pipeline(repo, tmp_path)
    only = (recipe_ids[0],)
    generate.generate.side_effect = GenerationError("rejected by the provider", False)
    assert runner.run(Options(ids=only, max_images=1)).exit_code == 1
    generate.generate.side_effect = None
    assert runner.run(Options(ids=only, max_images=1)).results == []  # nothing is retried unless asked
    assert runner.run(Options(ids=only, max_images=1, retry_failures=True)).exit_code == 0
    first, second = [r for r in rows(engine) if r.resource_id == recipe_ids[0]]
    assert (first.stage, second.stage) == ("failed", "completed")
    assert second.retry_of_attempt_id == first.id and second.attempt_number == 2 and second.retry_count == 0
    assert describe.describe.call_count == 1 and second.description == first.description  # description reused
    assert second.text_usage is None and second.image_usage is not None  # no text call, so no text usage


def test_uncertain_generation_is_never_retried_and_must_be_released_first(engine, repo, recipe_ids, tmp_path):
    runner, describe, generate, upload = pipeline(repo, tmp_path)
    recipe, only = recipe_ids[0], (recipe_ids[0],)
    generate.generate.side_effect = GenerationError("timed out after dispatch", True)
    first = runner.run(Options(ids=only, max_images=1))
    assert [r.status for r in first.results] == ["generation_uncertain"]
    generate.generate.side_effect = None
    retried = runner.run(Options(ids=only, max_images=1, retry_failures=True))
    assert retried.results == [] and generate.generate.call_count == 1  # reserved: not retried, not re-selected
    redo = runner.run(Options(redo=recipe, max_images=1))
    assert redo.results[0].status == "failed" and "may have been billed" in redo.results[0].detail
    assert runner.run(Options(release_uncertain=recipe)).results[0].status == "released"
    again = runner.run(Options(redo=recipe, max_images=1))
    assert again.exit_code == 0 and generate.generate.call_count == 2
    assert [r.stage for r in rows(engine) if r.resource_id == recipe] == ["abandoned", "completed"]


def test_release_uncertain_with_nothing_to_release_reports_failure(repo, recipe_ids, tmp_path):
    runner, *_ = pipeline(repo, tmp_path)
    result = runner.run(Options(release_uncertain=recipe_ids[0]))
    assert result.results[0].status == "failed" and result.exit_code == 1


def test_redo_through_the_orchestrator_reuses_the_description_and_costs_no_text(engine, repo, recipe_ids, tmp_path):
    runner, describe, generate, upload = pipeline(repo, tmp_path)
    first = runner.run(Options(ids=(recipe_ids[0],), max_images=1))
    recipe = first.results[0].resource_id
    runner.run(Options(redo=recipe, note="try again", max_images=1))
    old, new = [r for r in rows(engine) if r.resource_id == recipe]
    assert (old.stage, new.stage) == ("completed", "completed")
    assert new.retry_of_attempt_id == old.id and new.redo_note == "try again"
    assert describe.describe.call_count == 1 and new.text_usage is None
    with engine.connect() as conn:
        costs = conn.execute(text("SELECT attempt_id, total_tokens FROM catalog_image_generation_costs")).all()
    assert {c.attempt_id: c.total_tokens for c in costs} == {old.id: 3209, new.id: 2009}


def test_retry_failures_stops_after_four_attempts_in_postgres(engine, repo, recipe_ids):
    from forkplan_images.state import MAX_ATTEMPTS
    previous = None
    for number in range(1, MAX_ATTEMPTS + 1):
        previous = insert_attempt(engine, recipe_ids[0], number=number, stage="failed", retry_of_attempt_id=previous)
        candidates = repo.candidates(10, (recipe_ids[0],), True)
        assert bool(candidates) == (number < MAX_ATTEMPTS)
    assert repo.redo(recipe_ids[0], "admin", 60, None).attempt.attempt_number == MAX_ATTEMPTS + 1
