import base64
from typing import Protocol

import httpx
from openai import APIStatusError, OpenAI

from .network import IMAGE_TIMEOUT_SECONDS, Checkpoint, request
from .pricing import build_usage
from .state import LeaseLost


def is_definite_rejection(status_code: int) -> bool:
    """A 4xx (other than 408) was refused before generation, so nothing was billed. Anything else may have been."""
    return 400 <= status_code < 500 and status_code != 408


class GenerationError(RuntimeError):
    def __init__(self, message: str, uncertain: bool):
        super().__init__(message)
        self.uncertain = uncertain


class Generator(Protocol):
    def generate(self, prompt: str, settings: dict, *, checkpoint: Checkpoint | None = None) -> bytes: ...


class OpenAIGenerator:
    def __init__(self, client: OpenAI, downloads: httpx.Client):
        self.client = client.with_options(max_retries=0, timeout=IMAGE_TIMEOUT_SECONDS)
        self.downloads = downloads
        self.last_usage: dict | None = None

    def generate(self, prompt: str, settings: dict, *, checkpoint: Checkpoint | None = None) -> bytes:
        try:
            response = self.client.images.generate(
                model=settings["image_model"], prompt=prompt, n=1,
                quality=settings["quality"], size=settings["size"], output_format="png",
            )
        except APIStatusError as exc:
            raise GenerationError(str(exc), uncertain=not is_definite_rejection(exc.status_code)) from None
        except Exception as exc:
            # Connection errors and unknown failures may follow a billed dispatch.
            raise GenerationError(str(exc), uncertain=True) from None
        usage = getattr(response, "usage", None)
        self.last_usage = build_usage(
            "openai", settings["image_model"], input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_image_tokens=getattr(usage, "output_tokens", 0) or 0) if usage else None
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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiGenerator:
    """Gemini image models via generateContent. One request only: never retried automatically."""

    def __init__(self, client: httpx.Client, api_key: str):
        self.client, self.api_key = client, api_key
        self.last_usage: dict | None = None

    def generate(self, prompt: str, settings: dict, *, checkpoint: Checkpoint | None = None) -> bytes:
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"],
                                 "imageConfig": {"aspectRatio": "1:1"}},
        }
        try:
            response = self.client.post(
                GEMINI_URL.format(model=settings["image_model"]), json=body,
                headers={"x-goog-api-key": self.api_key}, timeout=IMAGE_TIMEOUT_SECONDS)
        except Exception as exc:
            raise GenerationError(type(exc).__name__, uncertain=True) from None
        if response.status_code != 200:
            raise GenerationError(f"Gemini HTTP {response.status_code}: {response.text[:300]}",
                                  uncertain=not is_definite_rejection(response.status_code))
        try:
            payload = response.json()
        except Exception:
            raise GenerationError("Gemini response was not JSON", uncertain=True) from None
        usage_metadata = payload.get("usageMetadata") or {}
        image_tokens = sum(d.get("tokenCount", 0) for d in usage_metadata.get("candidatesTokensDetails") or []
                           if d.get("modality") == "IMAGE")
        output_tokens = (usage_metadata.get("candidatesTokenCount") or 0) + (usage_metadata.get("thoughtsTokenCount") or 0)
        self.last_usage = build_usage(
            "gemini", settings["image_model"], input_tokens=usage_metadata.get("promptTokenCount") or 0,
            output_text_tokens=max(output_tokens - image_tokens, 0),
            output_image_tokens=image_tokens) if usage_metadata else None
        blocked = (payload.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            # Rejected before generation: definite, not billed as an image.
            raise GenerationError(f"Gemini blocked the prompt: {blocked}", uncertain=False)
        try:
            for part in payload["candidates"][0]["content"]["parts"]:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    data = base64.b64decode(inline["data"], validate=True)
                    if data:
                        return data
            reason = payload["candidates"][0].get("finishReason", "unknown")
        except Exception:
            raise GenerationError("Gemini response could not be decoded", uncertain=True) from None
        raise GenerationError(f"Gemini returned no image (finishReason={reason})", uncertain=True)
