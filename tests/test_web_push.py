"""WS-D.3 — Web Push module + endpoint tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.config import router as config_router
from app.api.push import router as push_router

# pywebpush import is lazy inside _send_blocking; we patch where it's used.
PYWEBPUSH_TARGET = "app.push.web_push.webpush"
PYWEBPUSH_EXC = "app.push.web_push.WebPushException"


# ── module: send_push ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_push_returns_sent_on_success(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "fake-priv")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "fake-pub")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:t@example.com")

    from app.push import web_push as wp

    with patch("app.push.web_push._send_blocking", return_value="sent"):
        outcome = await wp.send_push(
            {"endpoint": "https://x", "keys": {"p256dh": "k", "auth": "a"}},
            {"title": "t", "body": "b", "slot": "lunch", "url": "/demo.html"},
        )
    assert outcome == "sent"


@pytest.mark.asyncio
async def test_send_push_raises_when_disabled(monkeypatch):
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    from app.push import web_push as wp
    from app.push.web_push import WebPushDisabled
    with pytest.raises(WebPushDisabled):
        await wp.send_push({"endpoint": "x", "keys": {"p256dh": "k", "auth": "a"}},
                           {"title": "t", "body": "b", "slot": "lunch"})


def test_is_configured_requires_all_three(monkeypatch):
    from app.push.web_push import is_configured
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.delenv("VAPID_SUBJECT", raising=False)
    assert is_configured() is False
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")
    assert is_configured() is True


def test_get_public_key_returns_value_when_set(monkeypatch):
    from app.push.web_push import get_public_key
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "BMTk7L8")
    assert get_public_key() == "BMTk7L8"
    monkeypatch.delenv("VAPID_PUBLIC_KEY")
    assert get_public_key() is None


def test_send_blocking_410_returns_expired(monkeypatch):
    """410 Gone → 'expired' so the caller can clear the row.

    pywebpush is imported lazily inside _send_blocking, so we patch the
    upstream module — the lazy import resolves to our patched attrs.
    """
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")
    from app.push import web_push as wp

    class FakeResp:
        status_code = 410

    class FakeExc(Exception):
        response = FakeResp()

    def boom(*a, **kw):
        raise FakeExc("gone")

    with patch("pywebpush.WebPushException", FakeExc), \
         patch("pywebpush.webpush", side_effect=boom):
        outcome = wp._send_blocking(
            {"endpoint": "x", "keys": {"p256dh": "k", "auth": "a"}},
            {"title": "t", "body": "b", "slot": "x"},
        )
    assert outcome == "expired"


def test_build_payload_shape():
    from app.push.web_push import build_payload
    p = build_payload("lunch", "Hola", "¿Ya almorzaste?", url="/x")
    assert p == {"title": "Hola", "body": "¿Ya almorzaste?", "slot": "lunch", "url": "/x"}


# ── /api/v1/config — exposes vapid_public_key conditionally ──────────────────


@pytest.mark.asyncio
async def test_config_includes_vapid_public_key_when_set(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "fake-priv")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "BMTk7L8VAPID")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:t@x")

    app = FastAPI()
    app.include_router(config_router, prefix="/api/v1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/config")
    assert resp.status_code == 200
    assert resp.json().get("vapid_public_key") == "BMTk7L8VAPID"


@pytest.mark.asyncio
async def test_config_omits_vapid_public_key_when_unset(monkeypatch):
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("VAPID_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("VAPID_SUBJECT", raising=False)

    app = FastAPI()
    app.include_router(config_router, prefix="/api/v1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/config")
    assert resp.status_code == 200
    assert "vapid_public_key" not in resp.json()


# ── endpoints: POST + DELETE /users/push-subscription ────────────────────────


_FAKE_CLAIMS = {
    "sub": "1234567890",
    "email": "u@x.com",
    "email_verified": True,
}

VALID_SUB = {
    "endpoint": "https://fcm.googleapis.com/fcm/send/abc",
    "keys": {"p256dh": "BHxk7L8uvw", "auth": "abcdefgh"},
}


def _push_app():
    app = FastAPI()
    app.include_router(push_router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_post_subscription_503_when_disabled(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    app = _push_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/push-subscription", json={
            "id_token": "x", "subscription": VALID_SUB,
        })
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_post_subscription_invalid_id_token_returns_401(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")

    app = _push_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               side_effect=ValueError("expired")):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/users/push-subscription", json={
                "id_token": "fake", "subscription": VALID_SUB,
            })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_post_subscription_malformed_returns_422(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")

    app = _push_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/users/push-subscription", json={
                "id_token": "fake",
                "subscription": {"endpoint": "x"},  # missing keys
            })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_subscription_success_calls_db_update(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")

    app = _push_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS), \
         patch("app.api.push._update_user_push", new=AsyncMock(return_value=True)) as mu:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/users/push-subscription", json={
                "id_token": "fake", "subscription": VALID_SUB,
            })
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mu.assert_awaited_once()
    args = mu.call_args.args
    assert args[0] == "google_1234567890"
    assert args[1] == VALID_SUB


@pytest.mark.asyncio
async def test_post_subscription_user_not_found_returns_404(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:x@y")

    app = _push_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS), \
         patch("app.api.push._update_user_push", new=AsyncMock(return_value=False)):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/users/push-subscription", json={
                "id_token": "fake", "subscription": VALID_SUB,
            })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_subscription_idempotent_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    app = _push_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS), \
         patch("app.api.push._update_user_push", new=AsyncMock(return_value=False)) as mu:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.request("DELETE", "/api/v1/users/push-subscription",
                                    json={"id_token": "fake"})
    assert resp.status_code == 200
    mu.assert_awaited_once_with("google_1234567890", None)
