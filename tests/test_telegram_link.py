"""Tests for the Telegram link-token endpoint.

Mocks the Redis client (in-memory dict) so the suite stays self-
contained. Contract under test:
  · 503 when TELEGRAM_BOT_USERNAME env var is unset.
  · 400 when user_id is anonymous (UUID, not google_<sub> / tg_<id>).
  · 200 + 16-char hex token + deep_link contains the username and the
    "alma_<token>" start parameter.
  · The token is stored in Redis under the right key with a 10-min TTL
    pointing at the user_id.
  · Two consecutive calls produce DIFFERENT tokens (no collision).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.telegram_link import router as tg_router


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex


def _make_app(fake_redis: _FakeRedis | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(tg_router, prefix="/api/v1")
    app.state.scheduler_redis = fake_redis or _FakeRedis()
    return app


@pytest.mark.asyncio
async def test_returns_503_when_bot_username_unset(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_USERNAME", raising=False)
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/users/telegram-link/token", json={"user_id": "google_42"}
        )
    assert resp.status_code == 503
    assert "TELEGRAM_BOT_USERNAME" in resp.text


@pytest.mark.asyncio
async def test_anonymous_uuid_rejected(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "TestBot")
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/users/telegram-link/token", json={"user_id": "abc-uuid"}
        )
    assert resp.status_code == 400
    assert "authenticated identity" in resp.text


@pytest.mark.asyncio
async def test_happy_path_returns_token_and_link(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "AlmaHackathonBot")
    fake = _FakeRedis()
    app = _make_app(fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/users/telegram-link/token", json={"user_id": "google_42"}
        )
    assert resp.status_code == 200
    body = resp.json()
    token = body["token"]
    assert len(token) == 16  # 8 bytes hex
    assert all(ch in "0123456789abcdef" for ch in token)
    assert body["deep_link"] == f"https://t.me/AlmaHackathonBot?start=alma_{token}"

    # Stored in Redis under expected key with TTL
    assert fake.store[f"alma:tg-link:{token}"] == "google_42"
    assert fake.ttls[f"alma:tg-link:{token}"] == 600


@pytest.mark.asyncio
async def test_at_prefix_in_username_is_stripped(monkeypatch):
    """If someone configures TELEGRAM_BOT_USERNAME=@MyBot we should not
    end up with a https://t.me/@MyBot link (Telegram returns 400)."""
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "@AlmaHackathonBot")
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/users/telegram-link/token", json={"user_id": "google_42"}
        )
    assert resp.status_code == 200
    assert "https://t.me/AlmaHackathonBot?start=alma_" in resp.json()["deep_link"]


@pytest.mark.asyncio
async def test_consecutive_calls_produce_different_tokens(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "TestBot")
    fake = _FakeRedis()
    app = _make_app(fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        a = (await c.post("/api/v1/users/telegram-link/token", json={"user_id": "google_42"})).json()
        b = (await c.post("/api/v1/users/telegram-link/token", json={"user_id": "google_42"})).json()
    assert a["token"] != b["token"]
    # Both stored, both pointing at the same user
    assert fake.store[f"alma:tg-link:{a['token']}"] == "google_42"
    assert fake.store[f"alma:tg-link:{b['token']}"] == "google_42"


@pytest.mark.asyncio
async def test_telegram_user_id_also_accepted(monkeypatch):
    """A Telegram-native user (tg_<chat_id>) requesting their own link
    token is also valid — they want to add web auth later."""
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "TestBot")
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/users/telegram-link/token", json={"user_id": "tg_98765"}
        )
    assert resp.status_code == 200
