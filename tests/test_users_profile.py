"""Tests for the user profile / preferences endpoint.

Mocks the MCP client so the suite runs without a live MCP. Covers:

  · 400 if user_id is anonymous (UUID, not google_<sub> or tg_<id>).
  · 422 if age_range / proactive_channel are outside the allowed set.
  · NEW (E.8) — feature opt-ins:
      remember_consent (bool), proactive_channel ("none"|"push"|"telegram"|"sms").
      Both fields are optional; we only upsert the row when explicitly set.
      "none" channel IS a valid explicit choice ("don't contact me").
  · LEGACY (WS-D.5) — age_range + phone still accepted as optional fields
    so any deployed client keeps working. Phone < 7 digits silently dropped.
  · Idempotent on repeat submission (same entry_keys collapse server-side
    via UNIQUE constraint).
"""
from __future__ import annotations

from typing import Any

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


# ──────────────────────────── Identity / validation ────────────────────────────


@pytest.mark.asyncio
async def test_anonymous_uuid_rejected():
    app = _make_app()
    payload = {"user_id": "abc-123-uuid", "name": "Cristian"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 400
    assert "authenticated identity" in resp.text


@pytest.mark.asyncio
async def test_telegram_user_id_accepted():
    app = _make_app()
    payload = {"user_id": "tg_42", "name": "Cristian"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_name_whitespace_only_rejected():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "   "}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 422
    assert len(fake.calls) == 0


@pytest.mark.asyncio
async def test_invalid_proactive_channel_returns_422():
    """Unsupported channel values must fail Pydantic Literal validation."""
    app = _make_app()
    payload = {"user_id": "google_999", "name": "Cristian", "proactive_channel": "fax"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 422


# ──────────────────────────── New feature opt-ins (E.8) ────────────────────────


@pytest.mark.asyncio
async def test_minimal_payload_just_writes_name():
    """Submitting only user_id + name (no opt-ins, no legacy fields) must
    succeed and write exactly one row — the canonical name. This is the
    smallest valid call after the E.8 reframe."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(fake.calls) == 1
    assert fake.calls[0]["data"]["entry_key"] == "name"
    assert fake.calls[0]["data"]["name"] == "Cristian"


@pytest.mark.asyncio
async def test_remember_consent_true_persists_row():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "remember_consent": True}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)

    assert resp.status_code == 200
    by_key = {c["data"]["entry_key"]: c["data"] for c in fake.calls}
    assert "remember_consent" in by_key
    assert by_key["remember_consent"]["remember"] is True


@pytest.mark.asyncio
async def test_remember_consent_false_persists_row():
    """An explicit False is meaningfully different from omission — the
    user said 'no, don't remember', that's a stored decision."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "remember_consent": False}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    by_key = {c["data"]["entry_key"]: c["data"] for c in fake.calls}
    assert by_key["remember_consent"]["remember"] is False


@pytest.mark.asyncio
async def test_proactive_channel_none_is_explicit_optout():
    """'none' is a valid choice — the scheduler reads this row to know
    NOT to bother the user. It must be stored, not dropped."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "proactive_channel": "none"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    by_key = {c["data"]["entry_key"]: c["data"] for c in fake.calls}
    assert by_key["proactive_channel"]["channel"] == "none"


@pytest.mark.asyncio
async def test_proactive_channel_push_persists_row():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "proactive_channel": "push"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    by_key = {c["data"]["entry_key"]: c["data"] for c in fake.calls}
    assert by_key["proactive_channel"]["channel"] == "push"


@pytest.mark.asyncio
async def test_full_new_payload_writes_all_three():
    """Full feature-opt-in submission: name + remember + channel."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian Lazo",
        "remember_consent": True,
        "proactive_channel": "telegram",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    keys = {c["data"]["entry_key"] for c in fake.calls}
    assert keys == {"name", "remember_consent", "proactive_channel"}


# ──────────────────────────── Legacy WS-D.5 backward compat ────────────────────


@pytest.mark.asyncio
async def test_legacy_age_range_still_accepted():
    """Old WS-D.5 clients submitting age_range must keep working."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "age_range": "25-34"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    keys = {c["data"]["entry_key"] for c in fake.calls}
    assert {"name", "age_range"}.issubset(keys)


@pytest.mark.asyncio
async def test_legacy_phone_normalized_and_persisted():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "phone": "+51 999 888 777",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    by_key = {c["data"]["entry_key"]: c["data"] for c in fake.calls}
    assert by_key["phone"]["phone"] == "+51999888777"
    assert by_key["phone"]["phone_last4"] == "8777"


@pytest.mark.asyncio
async def test_legacy_phone_too_short_silently_dropped():
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {"user_id": "google_42", "name": "Cristian", "phone": "12345"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 200
    keys = {c["data"]["entry_key"] for c in fake.calls}
    assert "phone" not in keys


@pytest.mark.asyncio
async def test_invalid_age_range_returns_422():
    """Age_range typos hit the Literal validator before the body handler."""
    app = _make_app()
    payload = {"user_id": "google_999", "name": "Cristian", "age_range": "kid"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/users/profile", json=payload)
    assert resp.status_code == 422


# ──────────────────────────── Idempotency ──────────────────────────────────────


@pytest.mark.asyncio
async def test_idempotent_double_call_uses_same_entry_keys():
    """Submitting twice (e.g. the user reopens the modal to change channel)
    targets the same entry_keys both times — MCP's UNIQUE constraint
    collapses them into one row per (user_id, layer, entry_key)."""
    fake = _FakeMCP()
    app = _make_app(fake)
    payload = {
        "user_id": "google_42",
        "name": "Cristian",
        "remember_consent": True,
        "proactive_channel": "push",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/api/v1/users/profile", json=payload)
        # User changes their mind — switch channel
        payload["proactive_channel"] = "telegram"
        await c.post("/api/v1/users/profile", json=payload)

    channel_calls = [c for c in fake.calls if c["data"]["entry_key"] == "proactive_channel"]
    assert len(channel_calls) == 2
    assert channel_calls[0]["data"]["channel"] == "push"
    assert channel_calls[1]["data"]["channel"] == "telegram"
