"""WS-D.2 — tests for /auth/link-anonymous + /auth/google extension."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.auth import router as auth_router


def _make_app(memory: dict | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")
    mcp = MagicMock()
    mcp.get_memory = AsyncMock(return_value=memory or {
        "mood_history": [], "mentioned_events": [],
        "habits": [], "interaction_prefs": [],
    })
    mcp.link_anonymous_to_account = AsyncMock(
        return_value={"ok": True, "action": "keep", "merged_layers": 4}
    )
    app.state.mcp_client = mcp
    return app


_FAKE_CLAIMS = {
    "sub": "1234567890",
    "email": "user@example.com",
    "email_verified": True,
    "name": "Test User",
}


# ── /auth/link-anonymous ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_link_keep_returns_200_and_calls_mcp_keep(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
                "action": "keep",
            })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"ok": True, "action": "keep", "merged_layers": 4}
    app.state.mcp_client.link_anonymous_to_account.assert_awaited_once_with(
        "uuid_abc12345-6789-0abc-def0-123456789abc",
        "google_1234567890",
        "keep",
    )


@pytest.mark.asyncio
async def test_post_link_reset_returns_200_and_calls_mcp_reset(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    app.state.mcp_client.link_anonymous_to_account = AsyncMock(
        return_value={"ok": True, "action": "reset", "merged_layers": 0}
    )
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
                "action": "reset",
            })

    assert resp.status_code == 200
    assert resp.json()["action"] == "reset"


@pytest.mark.asyncio
async def test_post_link_invalid_action_returns_422(monkeypatch):
    """Pydantic Literal['keep','reset'] rejects 'merge' at the validation layer."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
                "action": "merge",
            })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_link_demo_uuid_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "demo_mateo",
                "action": "keep",
            })
    assert resp.status_code == 400
    assert "Invalid anon_uuid" in resp.text or "reserved" in resp.text


@pytest.mark.asyncio
async def test_post_link_google_prefix_rejected(monkeypatch):
    """Trying to link a google_* identifier as if it were anonymous."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "google_xyz",
                "action": "keep",
            })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_post_link_invalid_id_token_returns_401(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               side_effect=ValueError("expired")):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/link-anonymous", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
                "action": "keep",
            })
    assert resp.status_code == 401


# ── /auth/google extension — prior_anon_session ──────────────────────────────


@pytest.mark.asyncio
async def test_google_verify_with_anon_low_msg_count_returns_no_prompt(monkeypatch):
    """anon has 2 entries (below the 3 threshold) → should_prompt=false."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app(memory={
        "mood_history": [{"content": {"mood": "ok", "crisis_score": 0}}],
        "mentioned_events": [{"content": {"event": "x"}}],
        "habits": [], "interaction_prefs": [],
    })
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
            })
    body = resp.json()
    assert body["prior_anon_session"]["should_prompt"] is False
    assert body["prior_anon_session"]["auto_kept"] is False
    assert body["prior_anon_session"]["message_count"] == 2


@pytest.mark.asyncio
async def test_google_verify_with_anon_high_crisis_score_auto_keeps(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app(memory={
        "mood_history": [
            {"content": {"mood": "very low", "crisis_score": 0.7}},
            {"content": {"mood": "ok", "crisis_score": 0.1}},
        ],
        "mentioned_events": [{"content": {"event": "x"}} for _ in range(5)],
        "habits": [], "interaction_prefs": [],
    })
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
            })
    body = resp.json()
    assert body["prior_anon_session"]["auto_kept"] is True
    assert body["prior_anon_session"]["should_prompt"] is False
    app.state.mcp_client.link_anonymous_to_account.assert_awaited_once_with(
        "uuid_abc12345-6789-0abc-def0-123456789abc",
        "google_1234567890",
        "keep",
    )


@pytest.mark.asyncio
async def test_google_verify_with_anon_normal_returns_prompt_signal(monkeypatch):
    """17 messages, crisis 0.1 → should_prompt=true, auto_kept=false."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app(memory={
        "mood_history": [
            {"content": {"mood": "ok", "crisis_score": 0.1}} for _ in range(7)
        ],
        "mentioned_events": [{"content": {"event": "x"}} for _ in range(10)],
        "habits": [], "interaction_prefs": [],
    })
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={
                "id_token": "fake",
                "anon_uuid": "uuid_abc12345-6789-0abc-def0-123456789abc",
            })
    body = resp.json()
    assert body["prior_anon_session"]["should_prompt"] is True
    assert body["prior_anon_session"]["auto_kept"] is False
    assert body["prior_anon_session"]["message_count"] == 17
    # Auto-keep was NOT called
    app.state.mcp_client.link_anonymous_to_account.assert_not_called()


@pytest.mark.asyncio
async def test_google_verify_without_anon_uuid_returns_no_prior_session(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-id")
    app = _make_app()
    with patch("app.api.auth.google_id_token.verify_oauth2_token",
               return_value=_FAKE_CLAIMS):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            resp = await c.post("/api/v1/auth/google", json={"id_token": "fake"})
    assert resp.json()["prior_anon_session"] is None
