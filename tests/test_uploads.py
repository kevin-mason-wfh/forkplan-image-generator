import json
from uuid import uuid4

import httpx
import pytest
import respx

from forkplan_images.network import request
from forkplan_images.uploads import HttpUploader


@respx.mock
def test_auth_boundary_exact_bytes_and_content_type():
    resource_id = uuid4()
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{resource_id}/image/presign").respond(
        200, json={"upload_url": "https://bucket.example.test/put?signature=placeholder",
                   "public_url": "https://cdn.example.test/image",
                   "image_key": "new-key", "content_type": "image/jpeg", "expires_in": 300})
    put = respx.put("https://bucket.example.test/put?signature=placeholder").respond(200)
    with httpx.Client(headers={"Authorization": "inherited-must-not-leak"}) as client:
        uploader = HttpUploader(client, "https://api.example.test", "api-placeholder")
        presign = uploader.presign(resource_id, "image.jpg", 5)
        uploader.upload(presign, b"exact")
    assert api.calls[0].request.headers["Authorization"] == "Bearer " + "api-placeholder"
    assert json.loads(api.calls[0].request.content) == {
        "filename": "image.jpg", "content_type": "image/jpeg", "content_length": 5}
    assert "authorization" not in put.calls[0].request.headers
    assert put.calls[0].request.content == b"exact"
    assert put.calls[0].request.headers["Content-Type"] == "image/jpeg"


@respx.mock
def test_bounded_retry_backoff_and_timeout():
    route = respx.put("https://bucket.example.test/image").mock(
        side_effect=[httpx.Response(503), httpx.Response(429), httpx.Response(200)])
    sleeps = []
    with httpx.Client() as client:
        request(client, "PUT", "https://bucket.example.test/image", sleep=sleeps.append)
    assert route.call_count == 3
    assert 1 <= sleeps[0] <= 1.5
    assert 2 <= sleeps[1] <= 2.5
    assert route.calls[0].request.extensions["timeout"]["write"] == 60


@respx.mock
def test_redirect_is_not_followed():
    route = respx.put("https://bucket.example.test/image").respond(
        307, headers={"Location": "https://other.example.test/image"})
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        request(client, "PUT", "https://bucket.example.test/image", sleep=lambda _: None)
    assert route.call_count == 1
