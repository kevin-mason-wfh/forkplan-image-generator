"""Where the upload API's bearer token comes from: a fixed token, or a Cognito service-account login."""
import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .network import REQUEST_TIMEOUT_SECONDS, retry_call

EXPIRY_MARGIN_SECONDS = 60  # renew a little early so a token never expires mid-request


class TokenProvider(Protocol):
    def token(self) -> str: ...
    def refresh(self) -> str:
        """A new token after the API rejected the current one."""
        ...


class StaticToken:
    def __init__(self, value: str):
        self.value = value

    def token(self) -> str:
        return self.value

    def refresh(self) -> str:
        return self.value  # nothing to renew: the caller sees the same token and stops


class LoginError(RuntimeError):
    """Login was refused. status_code 401 makes the run stop like any other rejected credential."""
    status_code = 401

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class CognitoSettings:
    region: str
    client_id: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    client_secret: str = field(default="", repr=False)
    token_use: str = "access"  # which token the backend expects: "access" or "id"


class CognitoLogin:
    """Logs in a Cognito service-account user (USER_PASSWORD_AUTH) and keeps the token until shortly before it expires.

    Re-logging in is simpler and safer than refresh tokens: there is no rotating secret to store or race over when
    several Lambdas run at once."""

    def __init__(self, client: httpx.Client, settings: CognitoSettings, *,
                 clock: Callable[[], float] = time.monotonic):
        self.client, self.settings, self.clock = client, settings, clock
        self._token = ""
        self._expires_at = 0.0

    def token(self) -> str:
        if not self._token or self.clock() >= self._expires_at:
            self._login()
        return self._token

    def refresh(self) -> str:
        self._login()
        return self._token

    def _secret_hash(self) -> str:
        message = (self.settings.username + self.settings.client_id).encode()
        digest = hmac.new(self.settings.client_secret.encode(), message, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _login(self) -> None:
        s = self.settings
        parameters = {"USERNAME": s.username, "PASSWORD": s.password}
        if s.client_secret:
            parameters["SECRET_HASH"] = self._secret_hash()
        body = json.dumps({"AuthFlow": "USER_PASSWORD_AUTH", "ClientId": s.client_id, "AuthParameters": parameters})

        def send() -> httpx.Response:
            response = self.client.post(
                f"https://cognito-idp.{s.region}.amazonaws.com/", content=body.encode(), timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/x-amz-json-1.1",
                         "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"})
            if response.status_code >= 500:
                response.raise_for_status()
            return response

        response = retry_call(send, lambda exc: isinstance(exc, httpx.TransportError | httpx.HTTPStatusError))
        data = response.json() if response.content else {}
        if response.status_code != 200:
            raise LoginError("Cognito login failed", f"{data.get('__type', 'error')}: {data.get('message', '')}".strip())
        result = data.get("AuthenticationResult")
        if not result:
            challenge = data.get("ChallengeName", "unknown")
            raise LoginError("Cognito login needs a further step", f"challenge {challenge}; the service account must "
                             "have a permanent password (aws cognito-idp admin-set-user-password --permanent)")
        self._token = result["IdToken" if s.token_use == "id" else "AccessToken"]
        self._expires_at = self.clock() + max(int(result.get("ExpiresIn", 3600)) - EXPIRY_MARGIN_SECONDS, 0)
