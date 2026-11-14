import json
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from .auth import StaticToken, TokenProvider
from .network import Checkpoint, request, require_https


class Presign(BaseModel):
    upload_url: str
    public_url: str
    image_key: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    expires_in: int = Field(gt=0)


class Uploader(Protocol):
    def presign(self, resource_id: UUID, filename: str, length: int, *,
                checkpoint: Checkpoint | None = None) -> Presign: ...
    def upload(self, presign: Presign, data: bytes, *, checkpoint: Checkpoint | None = None) -> None: ...


class HttpUploader:
    def __init__(self, client: httpx.Client, base_url: str, auth: TokenProvider | str):
        self.client = client
        self.base_url = require_https(base_url).rstrip("/")
        self.auth = StaticToken(auth) if isinstance(auth, str) else auth

    def presign(self, resource_id: UUID, filename: str, length: int, *,
                checkpoint: Checkpoint | None = None) -> Presign:
        def send(token: str) -> httpx.Response:
            return request(
                self.client, "POST",
                f"{self.base_url}/v1/catalog/recipes/{resource_id}/image/presign",
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                content=json.dumps({"filename": filename, "content_type": "image/jpeg",
                                    "content_length": length}).encode(),
                checkpoint=checkpoint,
            )

        token = self.auth.token()
        try:
            response = send(token)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 401:
                raise
            # An expired token: a 401 means the request was not processed, so one retry with a new token is safe.
            fresh = self.auth.refresh()
            if fresh == token:
                raise
            response = send(fresh)
        presign = Presign.model_validate(response.json())
        require_https(presign.upload_url)
        if presign.content_type != "image/jpeg":
            raise ValueError("Presign returned incompatible content type")
        return presign

    def upload(self, presign: Presign, data: bytes, *, checkpoint: Checkpoint | None = None) -> None:
        request(self.client, "PUT", presign.upload_url,
                headers={"Content-Type": presign.content_type}, content=data, checkpoint=checkpoint)
