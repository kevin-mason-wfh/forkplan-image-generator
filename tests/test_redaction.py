from forkplan_images.redaction import redact


def test_urls_bearer_and_credentials_redacted():
    values = [
        "Bearer " + "unconfigured-token",
        "https://bucket.example.test/key?X-Amz-Credential=placeholder&X-Amz-Signature=placeholder",
        "https://api.example.test/request",
        "postgresql+psycopg://" + "db.example.test/catalog",
        "sk-placeholder-not-a-real-key",
        "configured-placeholder",
    ]
    result = redact(" ".join(values), ("configured-placeholder",))
    for value in values:
        assert value not in result
    assert "unconfigured-token" not in result
    assert len(redact("x" * 3000)) == 2000
