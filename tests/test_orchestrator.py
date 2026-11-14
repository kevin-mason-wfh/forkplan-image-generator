import time
from copy import deepcopy
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fakes import MemoryRepository
from test_storage_imaging import png

from forkplan_images.generation import GenerationError
from forkplan_images.imaging import process
from forkplan_images.orchestrator import Heartbeat, Options, Orchestrator
from forkplan_images.repository import Work
from forkplan_images.state import ACTIVE_STAGES, Recipe, Stage
from forkplan_images.storage import LocalStorage
from forkplan_images.uploads import Presign


def setup(tmp_path, count=1):
    recipes = [Recipe(uuid4(), f"Soup {i}") for i in range(count)]
    repo = MemoryRepository(recipes, secrets=("secret-placeholder",))
    store = LocalStorage(tmp_path)
    describe, generate, upload = Mock(), Mock(), Mock()
    describe.describe.return_value = "A smooth golden soup in a ceramic bowl."
    generate.generate.return_value = png()
    upload.presign.side_effect = lambda *args, **kwargs: Presign(
        upload_url="https://bucket.example.test/put", public_url="https://cdn.example.test/image",
        image_key=f"key-{upload.presign.call_count}", content_type="image/jpeg", expires_in=300)
    runner = Orchestrator(repo, store, describe, generate, upload,
                          {"image_model": "configured", "quality": "low", "size": "1024x1024"},
                          secrets=("secret-placeholder",))
    return runner, repo, store, describe, generate, upload, recipes


def seed(runner, repo, store, recipe, stage):
    a = repo.claim(Work(recipe), "old", 300)
    a = repo.update(a.id, a.lease_token, stage=stage,
                    last_successful_stage=Stage.PROMPT_BUILT if stage == Stage.GENERATING else stage,
                    description="A smooth golden soup in a ceramic bowl.", final_prompt="saved prompt",
                    settings=runner.settings, image_key="saved-upload-key")
    if stage in (Stage.GENERATING, Stage.GENERATED, Stage.PROCESSED, Stage.PRESIGNED, Stage.UPLOADED):
        store.put(a.original_storage_key, png())
    if stage in (Stage.PROCESSED, Stage.PRESIGNED, Stage.UPLOADED):
        jpeg = process(png())
        store.put(a.jpeg_storage_key, jpeg.data)
        a = repo.update(a.id, a.lease_token,
                        jpeg_sha256=jpeg.sha256, jpeg_length=jpeg.length)
    repo.release(a.id, a.lease_token)
    return a


@pytest.mark.parametrize("stage", ACTIVE_STAGES)
def test_resume_every_active_stage(tmp_path, stage):
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path)
    a = seed(runner, repo, store, recipes[0], stage)
    result = runner.run(Options())
    assert result.exit_code == 0
    assert repo.attempts[a.id].stage == Stage.COMPLETED
    assert generate.generate.call_count == int(stage in (Stage.SELECTED, Stage.DESCRIBED, Stage.PROMPT_BUILT))
    if stage == Stage.UPLOADED:
        upload.presign.assert_not_called()
        upload.upload.assert_not_called()
        assert repo.attempts[a.id].image_key == "saved-upload-key"
    else:
        assert repo.attempts[a.id].image_key == "key-1"
        uploaded = [changes for op, changes in repo.writes
                    if op == "update" and changes.get("stage") == Stage.UPLOADED]
        assert len(uploaded) == 1
        assert uploaded[0]["image_key"] == "key-1"


def test_generating_without_original_is_uncertain(tmp_path):
    runner, repo, store, _, generate, upload, recipes = setup(tmp_path)
    a = repo.claim(Work(recipes[0]), "old", 300)
    repo.update(a.id, a.lease_token, stage=Stage.GENERATING, last_successful_stage=Stage.PROMPT_BUILT)
    repo.release(a.id, a.lease_token)
    result = runner.run(Options(retry_failures=True, max_images=0))
    assert result.exit_code == 1
    assert repo.attempts[a.id].stage == Stage.UNCERTAIN
    assert result.images_dispatched == 0
    generate.generate.assert_not_called()
    upload.presign.assert_not_called()
    assert not repo.candidates(10, include_failed=True)


def test_marker_committed_before_dispatch_and_original_before_generated(tmp_path):
    runner, repo, store, _, generate, _, _ = setup(tmp_path)

    def generated(*args, **kwargs):
        a = next(iter(repo.attempts.values()))
        assert a.stage == Stage.GENERATING
        return png()

    generate.generate.side_effect = generated
    original_update = repo.update

    def update(aid, token, **changes):
        if changes.get("stage") == Stage.GENERATED:
            assert store.exists(repo.attempts[aid].original_storage_key)
        return original_update(aid, token, **changes)

    repo.update = update
    assert runner.run(Options()).exit_code == 0


@pytest.mark.parametrize("uncertain", [False, True])
def test_failed_dispatches_count_toward_cap_and_others_untouched(tmp_path, uncertain):
    runner, repo, _, _, generate, _, _ = setup(tmp_path, 3)
    generate.generate.side_effect = GenerationError("rejected", uncertain)
    result = runner.run(Options(limit=3, max_images=1))
    assert result.images_dispatched == 1
    assert result.exit_code == 1
    assert len(repo.attempts) == 1
    assert generate.generate.call_count == 1


def test_limit_and_max_images_zero_leave_work_untouched(tmp_path):
    runner, repo, _, describe, generate, upload, _ = setup(tmp_path, 3)
    runner.run(Options(limit=0))
    runner.run(Options(max_images=0))
    assert repo.writes == []
    describe.describe.assert_not_called()
    generate.generate.assert_not_called()
    upload.presign.assert_not_called()
    result = runner.run(Options(limit=1, max_images=3))
    assert result.images_dispatched == 1
    assert len(repo.attempts) == 1


def test_resume_before_fresh_and_recovered_original_costs_no_dispatch(tmp_path):
    runner, repo, store, _, generate, _, recipes = setup(tmp_path, 2)
    a = seed(runner, repo, store, recipes[1], Stage.GENERATING)
    result = runner.run(Options(limit=1, max_images=0))
    assert result.images_dispatched == 0
    assert len(repo.attempts) == 1
    assert repo.attempts[a.id].stage == Stage.COMPLETED
    generate.generate.assert_not_called()


def test_upload_failure_reuses_jpeg_and_requests_new_key(tmp_path):
    runner, repo, store, _, generate, upload, _ = setup(tmp_path)
    upload.upload.side_effect = RuntimeError("PUT failed https://bucket.example.test/?sig=secret-placeholder")
    first = runner.run(Options())
    assert first.exit_code == 1
    a = next(iter(repo.attempts.values()))
    assert a.last_successful_stage == Stage.PRESIGNED
    assert "https://" not in a.error_message
    assert "secret-placeholder" not in a.error_message
    saved = store.get(a.jpeg_storage_key)
    upload.upload.side_effect = None
    second = runner.run(Options(retry_failures=True, max_images=0))
    assert second.exit_code == 0
    assert generate.generate.call_count == 1
    assert upload.presign.call_count == 2
    assert upload.upload.call_args.args[1] == saved
    assert repo.attempts[a.id].image_key == "key-2"


def test_jpeg_integrity_checked_before_presign(tmp_path):
    runner, repo, store, _, generate, upload, recipes = setup(tmp_path)
    a = seed(runner, repo, store, recipes[0], Stage.PRESIGNED)
    repo.attempts[a.id].jpeg_sha256 = "corrupted"
    assert runner.run(Options(max_images=0)).exit_code == 1
    upload.presign.assert_not_called()
    generate.generate.assert_not_called()


def test_crash_after_put_represigns_with_new_key(tmp_path):
    runner, repo, store, _, generate, upload, _ = setup(tmp_path)
    real_update = repo.update

    class Crash(BaseException):
        pass

    def crash(aid, token, **changes):
        if changes.get("stage") == Stage.UPLOADED:
            raise Crash()
        return real_update(aid, token, **changes)

    repo.update = crash
    with pytest.raises(Crash):
        runner.run(Options())
    a = next(iter(repo.attempts.values()))
    assert a.stage == Stage.PRESIGNED
    repo.update = real_update
    repo.release(a.id, a.lease_token)
    assert runner.run(Options(max_images=0)).exit_code == 0
    assert generate.generate.call_count == 1
    assert upload.upload.call_count == 2
    assert repo.attempts[a.id].image_key == "key-2"


def test_lost_token_stops_without_error_write_or_more_work(tmp_path):
    runner, repo, _, describe, generate, upload, _ = setup(tmp_path)

    def describe_and_steal(recipe, **kwargs):
        next(iter(repo.attempts.values())).lease_token = uuid4()
        return "A smooth golden soup in a ceramic bowl."

    describe.describe.side_effect = describe_and_steal
    result = runner.run(Options())
    assert result.results[0].status == "lease_lost"
    assert not [x for x in repo.writes if x[0] in ("update", "release")]
    generate.generate.assert_not_called()
    upload.presign.assert_not_called()


def test_heartbeat_renews_during_long_operation(tmp_path):
    runner, repo, _, _, _, _, recipes = setup(tmp_path)
    a = repo.claim(Work(recipes[0]), "worker", 1)
    with Heartbeat(repo, a, 1) as heartbeat:
        heartbeat.call(lambda: time.sleep(0.75))
    assert sum(op == "renew" for op, _ in repo.writes) >= 3


def test_dry_run_no_writes_or_uploads(tmp_path):
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path, 2)
    seed(runner, repo, store, recipes[1], Stage.PRESIGNED)
    before = deepcopy(repo.attempts)
    repo.writes.clear()
    assert runner.run(Options(dry_run=True)).exit_code == 0
    assert repo.attempts == before
    assert repo.writes == []
    assert describe.describe.call_count == 1
    generate.generate.assert_not_called()
    upload.presign.assert_not_called()
    upload.upload.assert_not_called()


def test_one_failure_does_not_stop_next_recipe(tmp_path):
    runner, repo, _, describe, _, _, _ = setup(tmp_path, 2)
    describe.describe.side_effect = [ValueError("invalid"), "A golden soup in a ceramic bowl."]
    result = runner.run(Options(limit=2, max_images=2))
    assert [r.status for r in result.results] == ["failed", "completed"]
    assert result.exit_code == 1


def test_redo_recipe_with_image_and_no_history(tmp_path):
    runner, repo, _, _, _, _, recipes = setup(tmp_path)
    repo.recipes[recipes[0].id] = Recipe(recipes[0].id, "Soup", image_key="existing")
    assert runner.run(Options()).results == []
    assert runner.run(Options(redo=recipes[0].id, note="manual")).exit_code == 0
    assert len(repo.attempts) == 1


def test_claim_conflict_is_not_a_failure(tmp_path):
    runner, repo, _, describe, _, _, recipes = setup(tmp_path)
    old_candidates = repo.candidates

    def candidates(*args):
        work = old_candidates(*args)
        repo.claim(work[0], "other", 300)
        return work

    repo.candidates = candidates
    result = runner.run(Options())
    assert result.results[0].status == "skipped"
    assert result.exit_code == 0
    describe.describe.assert_not_called()


def test_renewals_during_generation_and_upload(tmp_path):
    runner, repo, _, _, generate, upload, _ = setup(tmp_path)
    runner.lease_seconds = 1
    observed = []

    def long_operation(result):
        before = sum(op == "renew" for op, _ in repo.writes)
        time.sleep(0.75)
        observed.append(sum(op == "renew" for op, _ in repo.writes) - before)
        return result

    generate.generate.side_effect = lambda *args, **kwargs: long_operation(png())
    upload.upload.side_effect = lambda *args, **kwargs: long_operation(None)
    assert runner.run(Options()).exit_code == 0
    assert all(count >= 1 for count in observed)


def test_lost_lease_during_generation_does_not_write_original(tmp_path):
    runner, repo, store, _, generate, upload, _ = setup(tmp_path)
    runner.lease_seconds = 1

    def steal(*args, **kwargs):
        next(iter(repo.attempts.values())).lease_token = uuid4()
        time.sleep(0.75)
        return png()

    generate.generate.side_effect = steal
    result = runner.run(Options())
    assert result.results[0].status == "lease_lost"
    a = next(iter(repo.attempts.values()))
    assert not store.exists(a.original_storage_key)
    assert a.stage == Stage.GENERATING
    upload.presign.assert_not_called()


def test_batched_describer_prefetches_only_what_the_image_cap_allows(tmp_path):
    from forkplan_images.descriptions import BatchedDescriber
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path, 3)
    batched = Mock(spec=BatchedDescriber)
    batched.describe.return_value = "A smooth golden soup in a ceramic bowl."
    runner.describer = batched
    runner.run(Options(limit=3, max_images=2))
    (prefetched,), keywords = batched.prefetch.call_args
    assert len(prefetched) == 2 and keywords == {"persist": True}


def test_dry_run_of_a_retry_makes_no_text_call(tmp_path):
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path)
    a = repo.claim(Work(recipes[0]), "old", 300)
    repo.update(a.id, a.lease_token, stage=Stage.FAILED, last_successful_stage=Stage.PROMPT_BUILT,
                description="A described dish.", description_version="v")
    repo.release(a.id, a.lease_token)
    result = runner.run(Options(dry_run=True, retry_failures=True, ids=(recipes[0].id,)))
    assert result.results[0].status == "dry_run" and "new attempt" in result.results[0].detail
    describe.describe.assert_not_called()
    generate.generate.assert_not_called()


class _Rejected(Exception):
    status_code = 401


def _http_error(status):
    import httpx
    request = httpx.Request("POST", "https://api.example.test/presign")
    return httpx.HTTPStatusError("rejected", request=request, response=httpx.Response(status, request=request))


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_upload_credentials_stop_the_run_and_the_saved_image_is_uploaded_on_retry(tmp_path, status):
    runner, repo, store, _, generate, upload, recipes = setup(tmp_path, 3)
    good_presign = upload.presign.side_effect
    upload.presign.side_effect = _http_error(status)
    result = runner.run(Options(limit=3, max_images=3))
    assert generate.generate.call_count == 1  # the other recipes did not each pay for an image that cannot be uploaded
    assert len(result.results) == 1 and result.exit_code == 1
    assert "rejected the credentials" in result.stopped_because and str(status) in result.stopped_because
    assert len(repo.attempts) == 1  # the other recipes were not even claimed
    (a,) = repo.attempts.values()
    assert a.stage == Stage.FAILED and a.last_successful_stage == Stage.PROCESSED
    upload.presign.side_effect = good_presign  # the token is fixed
    retry = runner.run(Options(retry_failures=True, max_images=0, limit=1))
    assert retry.exit_code == 0 and repo.attempts[a.id].stage == Stage.COMPLETED
    assert generate.generate.call_count == 1  # the saved image was uploaded, no new image was generated


def test_rejected_text_credentials_stop_before_any_image_is_requested(tmp_path):
    runner, repo, _, describe, generate, _, _ = setup(tmp_path, 3)
    describe.describe.side_effect = _Rejected("Incorrect API key")
    result = runner.run(Options(limit=3, max_images=3))
    assert len(result.results) == 1 and "rejected the credentials" in result.stopped_because
    describe.describe.assert_called_once()
    generate.generate.assert_not_called()


def test_other_http_errors_do_not_stop_the_run(tmp_path):
    runner, repo, _, _, generate, upload, _ = setup(tmp_path, 3)
    upload.presign.side_effect = _http_error(404)
    result = runner.run(Options(limit=3, max_images=3))
    assert len(result.results) == 3 and result.stopped_because is None


def test_dry_run_stops_on_rejected_credentials(tmp_path):
    runner, _, _, describe, _, _, _ = setup(tmp_path, 3)
    describe.describe.side_effect = _Rejected("Incorrect API key")
    result = runner.run(Options(limit=3, dry_run=True))
    assert len(result.results) == 1 and result.stopped_because and result.exit_code == 1


def test_auth_failure_status_reads_sdk_http_and_wrapped_errors():
    from forkplan_images.network import auth_failure_status
    assert auth_failure_status(_Rejected()) == 401
    assert auth_failure_status(_http_error(403)) == 403
    assert auth_failure_status(_http_error(500)) is None and auth_failure_status(ValueError("x")) is None
    wrapped = RuntimeError("generation failed")
    wrapped.__cause__ = _http_error(401)
    assert auth_failure_status(wrapped) == 401
