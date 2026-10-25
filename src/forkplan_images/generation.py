import base64
from typing import Protocol

import httpx
from openai import APIStatusError, OpenAI

from .network import Checkpoint, request
from .state import LeaseLost


IMAGE_TIMEOUT = 180.0


class GenerationError(RuntimeError):
    def __init__(self, message: str, uncertain: bool):
        super().__init__(message)
        self.uncertain = uncertain


class Generator(Protocol):
    def generate(self, prompt: str, settings: dict, *, checkpoint: Checkpoint | None = None) -> bytes: ...


class OpenAIGenerator:
    def __init__(self, client: OpenAI, downloads: httpx.Client):
        self.client = client.with_options(max_retries=0, timeout=IMAGE_TIMEOUT)
        self.downloads = downloads

    def generate(self, prompt: str, settings: dict, *, checkpoint: Checkpoint | None = None) -> bytes:
        try:
            response = self.client.images.generate(
                model=settings["image_model"], prompt=prompt, n=1,
                quality=settings["quality"], size=settings["size"], output_format="png",
            )
        except APIStatusError as exc:
            uncertain = not (400 <= exc.status_code < 500 and exc.status_code != 408)
            raise GenerationError(str(exc), uncertain=uncertain) from None
        except Exception as exc:
            # Connection errors and unknown failures may follow a billed dispatch.
            raise GenerationError(str(exc), uncertain=True) from None
        try:
            image = response.data[0]
            if image.b64_json:
                data = base64.b64decode(image.b64_json, validate=True)
            elif image.url:
                data = request(self.downloads, "GET", image.url, checkpoint=checkpoint).content
            else:
                raise ValueError("Image response contained no image")
            if not data:
                raise ValueError("Image response was empty")
            return data
        except LeaseLost:
            raise
        except Exception as exc:
            # A failed download/decode must never cause another generation request.
            raise GenerationError(str(exc), uncertain=True) from None
