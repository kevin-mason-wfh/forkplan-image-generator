from uuid import uuid4

from forkplan_images.state import ACTIVE_STAGES, SUCCESSFUL_STAGES, Attempt, Stage


def test_stage_and_storage_invariants():
    attempt = Attempt(uuid4())
    assert len(ACTIVE_STAGES) == 8
    assert Stage.GENERATING not in SUCCESSFUL_STAGES
    assert Stage.UPLOADED in ACTIVE_STAGES
    assert str(attempt.id) in attempt.original_storage_key
    assert not attempt.original_storage_key.startswith("/")
