import base64

import httpx
import pytest
import respx
from test_cli_config import environment

from forkplan_images.config import Config
from forkplan_images.generation import GEMINI_URL, GeminiGenerator, GenerationError

SETTINGS = {"image_model": "gemini-3.1-flash-image"}
URL = GEMINI_URL.format(model=SETTINGS["image_model"])
PNG = b"\x89PNG\r\n\x1a\nfake"


def ok(data=PNG):
    return {"candidates": [{"content": {"parts": [
        {"text": "note"}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode()}}]}}]}


@respx.mock
def test_success_sends_key_in_header_only_and_returns_original_bytes():
    route = respx.post(URL).respond(200, json=ok())
    with httpx.Client() as client:
        assert GeminiGenerator(client, "secret-key").generate("prompt", SETTINGS) == PNG
    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert route.call_count == 1


@pytest.mark.parametrize("status,uncertain", [(400, False), (403, False), (429, False),
                                             (408, True), (500, True), (503, True)])
@respx.mock
def test_status_classification_and_single_request(status, uncertain):
    route = respx.post(URL).respond(status, json={"error": {"message": "x"}})
    with httpx.Client() as client, pytest.raises(GenerationError) as error:
        GeminiGenerator(client, "k").generate("prompt", SETTINGS)
    assert error.value.uncertain is uncertain and route.call_count == 1


@respx.mock
def test_transport_failure_is_uncertain():
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with httpx.Client() as client, pytest.raises(GenerationError) as error:
        GeminiGenerator(client, "k").generate("prompt", SETTINGS)
    assert error.value.uncertain is True


@respx.mock
def test_blocked_prompt_is_definite_and_missing_image_is_uncertain():
    respx.post(URL).respond(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with httpx.Client() as client, pytest.raises(GenerationError) as blocked:
        GeminiGenerator(client, "k").generate("prompt", SETTINGS)
    assert blocked.value.uncertain is False
    respx.post(URL).respond(200, json={"candidates": [{"content": {"parts": [{"text": "no"}]},
                                                       "finishReason": "STOP"}]})
    with httpx.Client() as client, pytest.raises(GenerationError) as missing:
        GeminiGenerator(client, "k").generate("prompt", SETTINGS)
    assert missing.value.uncertain is True


def test_gemini_configuration():
    env = {**environment(), "IMAGE_PROVIDER": "gemini", "GEMINI_API_KEY": "g-key",
           "GEMINI_IMAGE_MODEL": "gemini-3.1-flash-image"}
    env.pop("OPENAI_IMAGE_MODEL")
    config = Config.from_environment(env)
    assert config.image_provider == "gemini" and config.settings["image_provider"] == "gemini"
    assert "g-key" in config.secrets and "g-key" not in repr(config)
    for bad in ({"GEMINI_API_KEY": ""}, {"GEMINI_IMAGE_MODEL": "gpt-image-1.5"}, {"IMAGE_PROVIDER": "x"}):
        with pytest.raises(ValueError):
            Config.from_environment({**env, **bad})
