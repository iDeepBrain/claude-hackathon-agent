"""Tests for the Google OAuth verify endpoint.

Mocks the ``google_id_token.verify_oauth2_token`` call so the suite
runs without network and without a real Google project. The contract
under test is:
  · 503 when GOOGLE_OAUTH_CLIENT_ID is unset (fail closed).
  · 401 on any ValueError from the verify call (Google rejects).
  · 401 when verify returns a payload missing ``sub``.
  · 200 + canonical user_id on success.
  · The endpoint NEVER bypasses verification — even an "almost valid"
    id_token must go through ``verify_oauth2_token``.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.auth import router as auth_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_returns_503_when_client_id_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/auth/google", json={"id_token": "anything"})
    assert resp.status_code == 503
    assert "GOOGLE_OAUTH_CLIENT_ID" in resp.text


@pytest.mark.asyncio
async def test_returns_user_id_for_valid_token(monkeypatch):
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_ID", "test-id.apps.googleusercontent.com"
    )
    fake_claims = {
        "sub": "1234567890",
        "email": "user@example.com",
        "email_verified": True,
        "name": "Test User",
        "picture": "https://example.com/p.jpg",
    }
    with patch(
        "app.api.auth.google_id_token.verify_oauth2_token",
        return_value=fake_claims,
    ):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={"id_token": "fake-valid"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "google_1234567890"
    assert data["email"] == "user@example.com"
    assert data["email_verified"] is True
    assert data["name"] == "Test User"
    assert data["picture"] == "https://example.com/p.jpg"


@pytest.mark.asyncio
async def test_returns_401_when_verify_raises(monkeypatch):
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_ID", "test-id.apps.googleusercontent.com"
    )
    with patch(
        "app.api.auth.google_id_token.verify_oauth2_token",
        side_effect=ValueError("Token expired"),
    ):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={"id_token": "expired"})
    assert resp.status_code == 401
    assert "Token expired" in resp.text


@pytest.mark.asyncio
async def test_returns_401_when_token_missing_sub(monkeypatch):
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_ID", "test-id.apps.googleusercontent.com"
    )
    with patch(
        "app.api.auth.google_id_token.verify_oauth2_token",
        return_value={"email": "user@example.com"},  # no sub
    ):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={"id_token": "no-sub"})
    assert resp.status_code == 401
    assert "sub" in resp.text


@pytest.mark.asyncio
async def test_audience_passed_to_verify(monkeypatch):
    """The endpoint MUST pass our configured client_id as the expected
    audience — without it, an attacker could submit an id_token from
    any other Google OAuth app and pass verification."""
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_ID", "alma.apps.googleusercontent.com"
    )
    captured: dict = {}

    def fake_verify(token, request, audience):
        captured["audience"] = audience
        return {"sub": "x", "email": "e@e.com"}

    with patch(
        "app.api.auth.google_id_token.verify_oauth2_token",
        side_effect=fake_verify,
    ):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/api/v1/auth/google", json={"id_token": "x"})

    assert captured["audience"] == "alma.apps.googleusercontent.com"


@pytest.mark.asyncio
async def test_email_unverified_passes_through(monkeypatch):
    """An unverified-email Google account is still a valid identity
    (some users have email verification disabled). The endpoint
    surfaces email_verified=false, never silently treats it as true."""
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_ID", "test.apps.googleusercontent.com"
    )
    with patch(
        "app.api.auth.google_id_token.verify_oauth2_token",
        return_value={"sub": "u", "email": "u@e.com", "email_verified": False},
    ):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={"id_token": "x"})

    assert resp.status_code == 200
    assert resp.json()["email_verified"] is False
