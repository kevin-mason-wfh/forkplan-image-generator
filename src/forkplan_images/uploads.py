import json
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from .network import request, require_https


class Presign(BaseModel):
    upload_url: str
    public_url: str
    image_key: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    expires_in: int = Field(gt=0)


class Uploader(Protocol):
    def presign(self, resource_id: UUID, filename: str, length: int) -> Presign: ...
    def upload(self, presign: Presign, data: bytes) -> None: ...


class HttpUploader:
    def __init__(self, client: httpx.Client, base_url: str, token: str):
        self.client = client
        self.base_url = require_https(base_url).rstrip("/")
        self.token = token

    def presign(self, resource_id: UUID, filename: str, length: int) -> Presign:
        response = request(
            self.client, "POST",
            f"{self.base_url}/v1/catalog/recipes/{resource_id}/image/presign",
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            content=json.dumps({"filename": filename, "content_type": "image/jpeg",
                                "content_length": length}).encode(),
        )
        presign = Presign.model_validate(response.json())
        require_https(presign.upload_url)
        if presign.content_type != "image/jpeg":
            raise ValueError("Presign returned incompatible content type")
        return presign

    def upload(self, presign: Presign, data: bytes) -> None:
        request(self.client, "PUT", presign.upload_url,
                headers={"Content-Type": presign.content_type}, content=data)
