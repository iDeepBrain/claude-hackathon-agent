"""Tests for the user profile onboarding endpoint.

Mocks the MCP client (via ``app.state.mcp_client``) so the suite runs
without a live MCP. Contract under test:
  · 400 if user_id is anonymous (UUID, not google_<sub> or tg_<id>).
  · 422 if age_range is outside the allowed set (handled by Pydantic).
  · 200 + three upserts on success (name, age_range, phone).
  · 200 + two upserts when phone is omitted/empty (skip phone layer).
  · Phone with too-few digits is silently dropped (no upsert), endpoint
    still returns 200 — better than rejecting on an edge case the user
    can't easily fix in the modal.
  · Phone is normalized: non-digits stripped except the leading +,
    last4 stored as-is from the digit-only suffix.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.users import router as users_router


class _FakeMCP:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def upsert_memory(self, user_id: str, layer: str, data: dict) -> dict:
        self.calls.append({"user_id": user_id, "layer": layer, "data": data})
        return {"success": True, "id": str(len(self.calls))}


def _make_app(fake_mcp: _FakeMCP | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(users_router, prefix="/api/v1")
    app.state.mcp_client = fake_mcp or _FakeMCP()
    return app


@pytest.mark.asyncio
async def test_anonymous_uuid_rejected():
    app = _make_app()
    payload = {
        "user_id": "abc-123-uuid",
        "name": "Cristian",
        "age_range": "25-34",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 400
    assert "authenticated identity" in resp.text


@pytest.mark.asyncio
async def test_invalid_age_range_returns_422():
    app = _make_app()
    payload = {
        "user_id": "google_999",
        "name": "Cristian",
        "age_range": "kid",  # not in allowed set
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_happy_path_writes_three_layers():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "age_range": "25-34",
        "phone": "+51 999 888 777",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(fake.calls) == 3

    by_key = {c["data"]["entry_key"]: c for c in fake.calls}
    assert by_key["name"]["data"]["name"] == "Cristian"
    assert by_key["name"]["data"]["preference"].startswith("Se llama")
    assert by_key["age_range"]["data"]["age_range"] == "25-34"
    # Phone normalized: spaces stripped, leading + kept
    assert by_key["phone"]["data"]["phone"] == "+51999888777"
    assert by_key["phone"]["data"]["phone_last4"] == "8777"
    assert all(c["layer"] == "interaction_prefs" for c in fake.calls)
    assert all(c["user_id"] == "google_42" for c in fake.calls)


@pytest.mark.asyncio
async def test_phone_omitted_skips_phone_layer():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "age_range": "prefer_not_to_say",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)

    assert resp.status_code == 200
    keys = {c["data"]["entry_key"] for c in fake.calls}
    assert keys == {"name", "age_range"}


@pytest.mark.asyncio
async def test_phone_too_short_silently_dropped():
    """A 5-digit string isn't a phone — likely a typo. Drop the phone
    upsert quietly so the user's other fields still land. Returning
    422 here would force a re-edit in the modal; the user already
    consented to optional phone."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "age_range": "25-34",
        "phone": "12345",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)

    assert resp.status_code == 200
    keys = {c["data"]["entry_key"] for c in fake.calls}
    assert "phone" not in keys
    assert keys == {"name", "age_range"}


@pytest.mark.asyncio
async def test_telegram_user_id_accepted():
    """tg_<id> is also a verified identity (Telegram handler validates
    chat_id), so profile capture from a future Telegram onboarding
    flow shouldn't 400."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "tg_42",
        "name": "Cristian",
        "age_range": "25-34",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_name_whitespace_only_rejected():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "   ",
        "age_range": "25-34",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 422
    assert len(fake.calls) == 0


@pytest.mark.asyncio
async def test_idempotent_double_call_uses_same_entry_keys():
    """Calling the endpoint twice (e.g. user re-opens modal to fix
    a typo) produces identical entry_keys both times — the upsert in
    MCP collapses them into a single row per layer per key."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "age_range": "25-34",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/api/v1/users/profile", json=payload)
        # second call with corrected name
        payload["name"] = "Cristian Lazo"
        await c.post("/api/v1/users/profile", json=payload)

    name_calls = [c for c in fake.calls if c["data"]["entry_key"] == "name"]
    assert len(name_calls) == 2
    # Both targeted the same entry_key — MCP's UNIQUE constraint will
    # ensure only one row exists despite the two upsert calls.
    assert name_calls[0]["data"]["entry_key"] == name_calls[1]["data"]["entry_key"]
    assert name_calls[1]["data"]["name"] == "Cristian Lazo"
