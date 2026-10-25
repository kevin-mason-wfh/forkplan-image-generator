import base64
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI
import pytest
import respx

from forkplan_images.generation import GenerationError, OpenAIGenerator


SETTINGS = {"image_model": "gpt-image-1.5", "quality": "low", "size": "1024x1024"}


@pytest.mark.parametrize("status,uncertain", [(400, False), (401, False), (403, False),
                                             (422, False), (429, False), (408, True),
                                             (500, True), (503, True)])
def test_failure_classification(status, uncertain):
    client = Mock()
    client.with_options.return_value = client
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"))
    client.images.generate.side_effect = APIStatusError("rejected", response=response, body={})
    with httpx.Client() as downloads:
        generator = OpenAIGenerator(client, downloads)
        with pytest.raises(GenerationError) as error:
            generator.generate("prompt", SETTINGS)
    assert error.value.uncertain is uncertain
    assert client.images.generate.call_count == 1
    client.with_options.assert_called_once_with(max_retries=0, timeout=180.0)


@respx.mock
def test_real_sdk_does_not_retry_429():
    route = respx.post("https://api.openai.com/v1/images/generations").respond(429, json={"error": {"message": "rate limited"}})
    with OpenAI(api_key="test-placeholder") as sdk, httpx.Client() as downloads:
        with pytest.raises(GenerationError):
            OpenAIGenerator(sdk, downloads).generate("prompt", SETTINGS)
    assert route.call_count == 1


def test_connection_failure_uncertain():
    client = Mock()
    client.with_options.return_value = client
    client.images.generate.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"))
    with httpx.Client() as downloads:
        with pytest.raises(GenerationError) as error:
            OpenAIGenerator(client, downloads).generate("prompt", SETTINGS)
    assert error.value.uncertain


@respx.mock
def test_base64_and_url_originals():
    client = Mock()
    client.with_options.return_value = client
    client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(b"original").decode(), url=None)])
    with httpx.Client(headers={"Authorization": "must-not-leak"}) as downloads:
        generator = OpenAIGenerator(client, downloads)
        assert generator.generate("prompt", SETTINGS) == b"original"
        client.images.generate.return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="https://images.example.test/image")])
        route = respx.get("https://images.example.test/image").respond(200, content=b"downloaded")
        assert generator.generate("prompt", SETTINGS) == b"downloaded"
        assert "authorization" not in route.calls[0].request.headers
        assert route.calls[0].request.extensions["timeout"]["read"] == 60
