from forkplan_images.state import ACTIVE, SUCCESSFUL, Attempt, Stage
from uuid import uuid4


def test_stage_and_storage_invariants():
    attempt = Attempt(uuid4())
    assert len(ACTIVE) == 8
    assert Stage.GENERATING not in SUCCESSFUL
    assert Stage.UPLOADED in ACTIVE
    assert str(attempt.id) in attempt.original_storage_key
    assert not attempt.original_storage_key.startswith("/")
