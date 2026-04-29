"""WS-D.3 — Scheduler push dispatch tests.

Mocks _fetch_active_push_users + send_push + redis_client. Verifies the
gates (silence window, crisis, auto-pause, slot-already-sent) and the
outcome handling (sent / expired / failed).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.scheduler import proactive


def _user(user_id: str = "google_x", days_ago: int = 0) -> dict:
    return {
        "user_id": user_id,
        "push_subscription": {
            "endpoint": "https://x", "keys": {"p256dh": "k", "auth": "a"},
        },
        "last_seen_at": datetime.now(timezone.utc) - timedelta(days=days_ago),
    }


def _redis_mock(slot_sent: bool = False, last_activity_ts: float | None = None,
                crisis_score: float | None = None) -> MagicMock:
    """Configurable redis-async-like mock."""
    r = MagicMock()
    r.exists = AsyncMock(return_value=1 if slot_sent else 0)

    async def _get(key: str):
        key_s = key.decode() if isinstance(key, bytes) else key
        if "last_activity" in key_s and last_activity_ts is not None:
            return str(last_activity_ts)
        if "crisis" in key_s and crisis_score is not None:
            return str(crisis_score)
        return None

    r.get = AsyncMock(side_effect=_get)
    r.set = AsyncMock(return_value=True)
    return r


@pytest.mark.asyncio
async def test_skip_when_vapid_unconfigured(monkeypatch):
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    r = _redis_mock()
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user()])) as fetch_mock:
        await proactive.send_proactive_push("breakfast", r)
    fetch_mock.assert_not_called()  # short-circuited before DB hit


@pytest.mark.asyncio
async def test_send_path_calls_send_push_and_marks_slot(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock()
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user("google_user1")])), \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="sent")) as sp:
        await proactive.send_proactive_push("lunch", r)

    sp.assert_awaited_once()
    r.set.assert_awaited_once()
    args = r.set.await_args.args
    assert "alma:proactive:slot:google_user1:" in args[0]
    assert args[0].endswith(":lunch")


@pytest.mark.asyncio
async def test_skip_user_with_recent_activity(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    # Activity 5 minutes ago → within the 2h silence window
    r = _redis_mock(last_activity_ts=datetime.now().timestamp() - 300)
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user()])), \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="sent")) as sp:
        await proactive.send_proactive_push("lunch", r)

    sp.assert_not_called()


@pytest.mark.asyncio
async def test_skip_user_in_crisis(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock(crisis_score=0.8)
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user()])), \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="sent")) as sp:
        await proactive.send_proactive_push("lunch", r)

    sp.assert_not_called()


@pytest.mark.asyncio
async def test_skip_when_slot_already_sent(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock(slot_sent=True)
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user()])), \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="sent")) as sp:
        await proactive.send_proactive_push("lunch", r)

    sp.assert_not_called()


@pytest.mark.asyncio
async def test_auto_pause_inactive_user(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock()
    user = _user("google_ghost", days_ago=10)  # 10d > 7d threshold

    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[user])), \
         patch("app.scheduler.proactive._mark_push_disabled",
               new=AsyncMock()) as mark, \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="sent")) as sp:
        await proactive.send_proactive_push("lunch", r)

    mark.assert_awaited_once()
    sp.assert_not_called()


@pytest.mark.asyncio
async def test_410_outcome_clears_subscription(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock()
    user = _user("google_dead")

    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[user])), \
         patch("app.scheduler.proactive._clear_push_subscription",
               new=AsyncMock()) as clear, \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="expired")):
        await proactive.send_proactive_push("breakfast", r)

    clear.assert_awaited_once_with("google_dead")
    r.set.assert_not_called()  # don't mark slot when delivery failed


@pytest.mark.asyncio
async def test_failed_outcome_does_not_clear_or_mark(monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "x")
    monkeypatch.setenv("VAPID_SUBJECT", "mailto:a@b")

    r = _redis_mock()
    with patch("app.scheduler.proactive._fetch_active_push_users",
               new=AsyncMock(return_value=[_user()])), \
         patch("app.scheduler.proactive._clear_push_subscription",
               new=AsyncMock()) as clear, \
         patch("app.scheduler.proactive.send_push",
               new=AsyncMock(return_value="failed")):
        await proactive.send_proactive_push("breakfast", r)

    clear.assert_not_called()
    r.set.assert_not_called()


@pytest.mark.asyncio
async def test_create_scheduler_adds_telegram_and_push_jobs(monkeypatch):
    """One slot tick fires both the Telegram and the push dispatcher."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    redis_client = MagicMock()
    sched = proactive.create_scheduler(redis_client)
    job_ids = {j.id for j in sched.get_jobs()}
    # 3 slots × 2 channels = 6 jobs
    assert len(job_ids) == 6
    for slot in ("breakfast", "lunch", "dinner"):
        assert f"proactive_{slot}" in job_ids
        assert f"proactive_push_{slot}" in job_ids
