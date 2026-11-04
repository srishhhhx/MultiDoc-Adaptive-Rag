def test_admin_endpoints_disabled_without_key(client, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    for method, path in [
        ("post", "/api/clear-database"),
        ("post", "/api/nuclear-reset"),
        ("get", "/api/debug/collections"),
        ("post", "/api/metrics/clear"),
        ("get", "/api/metrics/summary"),
    ]:
        assert getattr(client, method)(path).status_code == 404, path


def test_admin_endpoints_need_correct_key(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    assert client.get("/api/debug/collections").status_code == 403
    assert client.get("/api/debug/collections", headers={"X-Admin-Key": "bad"}).status_code == 403
    assert client.get("/api/debug/collections", headers={"X-Admin-Key": "secret"}).status_code == 200


def test_upload_over_limit_rejected(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 1024)
    r = client.post("/api/upload", files={"file": ("big.txt", b"x" * 4096)})
    assert r.status_code == 413


def test_unsupported_extension_rejected(client):
    r = client.post("/api/upload", files={"file": ("evil.exe", b"MZ")})
    assert r.status_code == 400


def test_health_is_public(client):
    assert client.get("/api/health").status_code == 200
