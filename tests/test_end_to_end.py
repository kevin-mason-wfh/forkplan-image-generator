import base64
from dataclasses import replace
import json
import logging
from unittest.mock import Mock
from uuid import uuid4

import pytest
import respx
import httpx

from fakes import MemoryRepository
from forkplan_images import cli
from forkplan_images.config import Config
from forkplan_images.state import Recipe, Stage
from test_cli_config import environment
from test_storage_imaging import png


@pytest.mark.parametrize("dry_run,upload_failure", [(False, False), (True, False), (False, True)])
@respx.mock
def test_cli_with_real_sdk_and_mocked_http(tmp_path, monkeypatch, capsys, caplog,
                                         dry_run, upload_failure):
    recipe = Recipe(uuid4(), "Soup", ingredients=({"name": "Onion", "quantity": 2, "unit": "whole"},))
    config = replace(Config.from_environment(environment()), artifact_dir=tmp_path / "artifacts")
    repo = MemoryRepository([recipe], secrets=config.secrets)
    monkeypatch.setattr(cli.Config, "from_environment", lambda _: config)
    monkeypatch.setattr(cli, "create_engine", Mock())
    monkeypatch.setattr(cli, "PostgresRepository", lambda *_: repo)
    text = respx.post("https://api.openai.com/v1/responses").respond(200, json={
        "id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
        "model": config.text_model, "output": [{
            "type": "message", "id": "msg_test", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "annotations": [],
                         "text": json.dumps({"description": "A smooth golden soup in a ceramic bowl."})}],
        }],
    })
    images = respx.post("https://api.openai.com/v1/images/generations").respond(
        200, json={"created": 0, "data": [{"b64_json": base64.b64encode(png()).decode()}]})
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{recipe.id}/image/presign").respond(
        200, json={"upload_url": "https://bucket.example.test/upload?signature=private-url-placeholder",
                   "public_url": "https://cdn.example.test/image", "image_key": "new-key",
                   "content_type": "image/jpeg", "expires_in": 300})
    put = respx.put("https://bucket.example.test/upload?signature=private-url-placeholder").respond(
        403 if upload_failure else 200)
    with caplog.at_level(logging.DEBUG):
        code = cli.main(["--limit", "1"] + (["--dry-run"] if dry_run else []))
    assert code == int(upload_failure)
    assert text.call_count == 1
    if dry_run:
        assert images.call_count == api.call_count == put.call_count == 0
        assert repo.writes == []
        assert not config.artifact_dir.exists()
    else:
        assert images.call_count == api.call_count == put.call_count == 1
        attempt = next(iter(repo.attempts.values()))
        assert attempt.stage == (Stage.FAILED if upload_failure else Stage.COMPLETED)
        if upload_failure:
            assert "https://" not in attempt.error_message
            assert "private-url-placeholder" not in attempt.error_message
        assert "authorization" not in put.calls[0].request.headers
        assert api.calls[0].request.headers["Authorization"] == "Bearer " + config.api_token
    output = capsys.readouterr()
    for value in (config.api_token, config.openai_api_key, "private-url-placeholder",
                  "https://bucket.example.test"):
        assert value not in output.out + output.err + caplog.text


@respx.mock
def test_upload_retry_stops_when_another_worker_takes_ownership(tmp_path, monkeypatch):
    recipe = Recipe(uuid4(), "Soup")
    config = replace(Config.from_environment(environment()), artifact_dir=tmp_path / "artifacts")
    repo = MemoryRepository([recipe], secrets=config.secrets)
    monkeypatch.setattr(cli.Config, "from_environment", lambda _: config)
    monkeypatch.setattr(cli, "create_engine", Mock())
    monkeypatch.setattr(cli, "PostgresRepository", lambda *_: repo)
    respx.post("https://api.openai.com/v1/responses").respond(200, json={
        "id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
        "model": config.text_model, "output": [{
            "type": "message", "id": "msg_test", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "annotations": [],
                         "text": json.dumps({"description": "A smooth golden soup in a ceramic bowl."})}],
        }],
    })
    images = respx.post("https://api.openai.com/v1/images/generations").respond(
        200, json={"created": 0, "data": [{"b64_json": base64.b64encode(png()).decode()}]})
    respx.post(f"https://api.example.test/v1/catalog/recipes/{recipe.id}/image/presign").respond(
        200, json={"upload_url": "https://bucket.example.test/upload",
                   "public_url": "https://cdn.example.test/image", "image_key": "new-key",
                   "content_type": "image/jpeg", "expires_in": 300})

    def lose_ownership(request):
        next(iter(repo.attempts.values())).lease_token = uuid4()
        return httpx.Response(503)

    put = respx.put("https://bucket.example.test/upload").mock(side_effect=lose_ownership)
    assert cli.main([]) == 1
    assert put.call_count == images.call_count == 1
    attempt = next(iter(repo.attempts.values()))
    assert attempt.stage == Stage.PRESIGNED
    assert attempt.error_message is None
