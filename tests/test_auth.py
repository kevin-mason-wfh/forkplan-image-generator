import base64
import hashlib
import hmac
import json
from uuid import uuid4

import httpx
import pytest
import respx
from test_cli_config import environment
from test_orchestrator import setup

from forkplan_images.auth import CognitoLogin, CognitoSettings, LoginError, StaticToken, TokenWithLoginFallback
from forkplan_images.config import Config
from forkplan_images.orchestrator import Options
from forkplan_images.uploads import HttpUploader

COGNITO = "https://cognito-idp.eu-west-2.amazonaws.com/"
SETTINGS = CognitoSettings("eu-west-2", "client123", "svc@example.test", "pw-placeholder")


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def login_response(token="tok-1", expires=3600, key="AccessToken"):
    return httpx.Response(200, json={"AuthenticationResult": {key: token, "ExpiresIn": expires}})


@respx.mock
def test_login_request_shape_and_token_is_cached_until_shortly_before_expiry():
    route = respx.post(COGNITO).mock(side_effect=[login_response("tok-1"), login_response("tok-2")])
    clock = Clock()
    with httpx.Client() as client:
        auth = CognitoLogin(client, SETTINGS, clock=clock)
        assert auth.token() == "tok-1" and auth.token() == "tok-1" and route.call_count == 1
        sent = route.calls[0].request
        assert sent.headers["X-Amz-Target"] == "AWSCognitoIdentityProviderService.InitiateAuth"
        assert sent.headers["Content-Type"] == "application/x-amz-json-1.1"
        assert json.loads(sent.content) == {"AuthFlow": "USER_PASSWORD_AUTH", "ClientId": "client123",
                                            "AuthParameters": {"USERNAME": "svc@example.test",
                                                               "PASSWORD": "pw-placeholder"}}
        clock.now = 3600 - 61
        assert auth.token() == "tok-1"  # still valid
        clock.now = 3600 - 59
        assert auth.token() == "tok-2" and route.call_count == 2  # renewed before it could expire mid-request


@respx.mock
def test_client_secret_adds_the_secret_hash_and_id_token_can_be_selected():
    route = respx.post(COGNITO).respond(200, json={"AuthenticationResult": {"IdToken": "id-tok", "AccessToken": "acc"}})
    settings = CognitoSettings("eu-west-2", "client123", "svc@example.test", "pw", "the-secret", "id")
    with httpx.Client() as client:
        assert CognitoLogin(client, settings).token() == "id-tok"
    expected = base64.b64encode(hmac.new(b"the-secret", b"svc@example.testclient123", hashlib.sha256).digest()).decode()
    assert json.loads(route.calls[0].request.content)["AuthParameters"]["SECRET_HASH"] == expected


@pytest.mark.parametrize("response,hint", [
    (httpx.Response(400, json={"__type": "NotAuthorizedException", "message": "Incorrect username or password."}),
     "NotAuthorizedException"),
    (httpx.Response(400, json={"__type": "InvalidParameterException",
                               "message": "USER_PASSWORD_AUTH flow not enabled for this client"}), "not enabled"),
    (httpx.Response(200, json={"ChallengeName": "NEW_PASSWORD_REQUIRED"}), "permanent password"),
])
@respx.mock
def test_refused_logins_explain_why_and_never_include_the_password(response, hint):
    respx.post(COGNITO).mock(return_value=response)
    with httpx.Client() as client, pytest.raises(LoginError) as error:
        CognitoLogin(client, SETTINGS).token()
    assert hint in error.value.hint and "pw-placeholder" not in str(error.value) + error.value.hint


@respx.mock
def test_cognito_server_errors_are_retried():
    route = respx.post(COGNITO).mock(side_effect=[httpx.Response(503), login_response()])
    with httpx.Client() as client:
        assert CognitoLogin(client, SETTINGS).token() == "tok-1"
    assert route.call_count == 2


PRESIGN = {"upload_url": "https://bucket.example.test/put", "public_url": "https://cdn.example.test/i",
           "image_key": "k", "content_type": "image/jpeg", "expires_in": 300}


@respx.mock
def test_an_expired_token_is_replaced_once_and_the_presign_retried():
    resource_id = uuid4()
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{resource_id}/image/presign").mock(
        side_effect=[httpx.Response(401), httpx.Response(200, json=PRESIGN)])
    respx.post(COGNITO).mock(side_effect=[login_response("expired"), login_response("fresh")])
    with httpx.Client() as client:
        uploader = HttpUploader(client, "https://api.example.test", CognitoLogin(client, SETTINGS))
        assert uploader.presign(resource_id, "i.jpg", 5).image_key == "k"
    assert [c.request.headers["Authorization"] for c in api.calls] == ["Bearer expired", "Bearer fresh"]


@respx.mock
def test_a_second_401_with_a_fresh_token_is_not_retried_again():
    resource_id = uuid4()
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{resource_id}/image/presign").respond(401)
    respx.post(COGNITO).mock(side_effect=[login_response("a"), login_response("b")])
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        HttpUploader(client, "https://api.example.test", CognitoLogin(client, SETTINGS)).presign(resource_id, "i.jpg", 5)
    assert api.call_count == 2


@respx.mock
def test_a_fixed_token_is_not_retried_and_other_errors_do_not_refresh():
    resource_id = uuid4()
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{resource_id}/image/presign").respond(401)
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        HttpUploader(client, "https://api.example.test", StaticToken("fixed")).presign(resource_id, "i.jpg", 5)
    assert api.call_count == 1


def test_a_refused_login_stops_the_run_with_the_reason(tmp_path):
    runner, repo, _, _, generate, upload, _ = setup(tmp_path, 3)
    upload.presign.side_effect = LoginError("Cognito login failed", "NotAuthorizedException: Incorrect username or password.")
    result = runner.run(Options(limit=3, max_images=3))
    assert generate.generate.call_count == 1 and len(result.results) == 1
    assert "NotAuthorizedException" in result.stopped_because and "COGNITO_" in result.stopped_because


# -- configuration -------------------------------------------------------------------------------------------

def cognito_environment(**extra):
    env = {k: v for k, v in environment().items() if k != "API_TOKEN"}
    return {**env, "COGNITO_REGION": "eu-west-2", "COGNITO_CLIENT_ID": "client123",
            "COGNITO_USERNAME": "svc@example.test", "COGNITO_PASSWORD": "pw with spaces", **extra}


def test_cognito_mode_needs_no_api_token_and_hides_its_secrets():
    config = Config.from_environment(cognito_environment(COGNITO_CLIENT_SECRET="s3cret", COGNITO_TOKEN_USE="ID"))
    assert config.cognito.token_use == "id" and config.api_token == ""
    assert "pw with spaces" in config.secrets and "s3cret" in config.secrets
    assert "pw with spaces" not in repr(config)


@pytest.mark.parametrize("change,message", [
    ({"COGNITO_PASSWORD": ""}, "COGNITO_PASSWORD"),
    ({"COGNITO_REGION": "london"}, "COGNITO_REGION"),
    ({"COGNITO_TOKEN_USE": "refresh"}, "COGNITO_TOKEN_USE"),
])
def test_bad_cognito_settings_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        Config.from_environment(cognito_environment(**change))


def test_without_cognito_settings_the_api_token_is_still_required():
    env = {k: v for k, v in environment().items() if k != "API_TOKEN"}
    with pytest.raises(ValueError, match="API_TOKEN"):
        Config.from_environment(env)


@respx.mock
def test_fixed_token_is_used_first_and_the_login_takes_over_after_a_rejection():
    resource_id = uuid4()
    api = respx.post(f"https://api.example.test/v1/catalog/recipes/{resource_id}/image/presign").mock(
        side_effect=[httpx.Response(200, json=PRESIGN), httpx.Response(401), httpx.Response(200, json=PRESIGN),
                     httpx.Response(200, json=PRESIGN)])
    login = respx.post(COGNITO).mock(side_effect=[login_response("from-login")])
    with httpx.Client() as client:
        auth = TokenWithLoginFallback(StaticToken("fixed"), CognitoLogin(client, SETTINGS))
        uploader = HttpUploader(client, "https://api.example.test", auth)
        uploader.presign(resource_id, "a.jpg", 5)   # the fixed token works: no login at all
        assert login.call_count == 0
        uploader.presign(resource_id, "b.jpg", 5)   # rejected: sign in and retry
        uploader.presign(resource_id, "c.jpg", 5)   # stays on the login token, no second login
    assert [c.request.headers["Authorization"] for c in api.calls] == [
        "Bearer fixed", "Bearer fixed", "Bearer from-login", "Bearer from-login"]
    assert login.call_count == 1


def test_both_settings_are_accepted_together():
    config = Config.from_environment(cognito_environment(API_TOKEN="fixed-token"))
    assert config.api_token == "fixed-token" and config.cognito is not None
    assert "fixed-token" in config.secrets and "pw with spaces" in config.secrets


def test_cli_picks_the_right_auth_for_the_configuration():
    import httpx as _httpx

    from forkplan_images.cli import api_auth
    with _httpx.Client() as client:
        assert api_auth(client, Config.from_environment(environment())) == "placeholder-api"
        assert isinstance(api_auth(client, Config.from_environment(cognito_environment())), CognitoLogin)
        both = api_auth(client, Config.from_environment(cognito_environment(API_TOKEN="t")))
        assert isinstance(both, TokenWithLoginFallback)
