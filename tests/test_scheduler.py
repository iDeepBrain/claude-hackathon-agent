import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.scheduler.proactive import send_proactive, create_scheduler, SLOTS


@pytest.mark.asyncio
async def test_send_proactive_skips_already_sent():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    redis.get.side_effect = lambda k: b"999" if "alma:chat" in k else None
    redis.exists.return_value = True
    await send_proactive("breakfast", redis)
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_skips_crisis():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    redis.get.side_effect = lambda k: {
        "alma:chat:123": b"999",
        "alma:crisis:last:tg_123": b"0.8",
    }.get(k)
    redis.exists.return_value = False
    await send_proactive("breakfast", redis)
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_sends_when_gates_pass():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    redis.get.side_effect = lambda k: b"999" if k == "alma:chat:123" else None
    redis.exists.return_value = False

    with patch("app.scheduler.proactive.httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=mock_resp)))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        await send_proactive("breakfast", redis)

    redis.set.assert_called_once()


def test_create_scheduler_has_telegram_jobs_per_slot():
    """Telegram path: one job per slot. Web push adds another set; that
    set is asserted in test_scheduler_push.py."""
    redis = AsyncMock()
    scheduler = create_scheduler(redis)
    job_ids = {j.id for j in scheduler.get_jobs()}
    for slot in ("breakfast", "lunch", "dinner"):
        assert f"proactive_{slot}" in job_ids


@pytest.mark.asyncio
async def test_send_proactive_skips_recent_activity():
    import time
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    recent_ts = str(time.time() - 60).encode()
    redis.get.side_effect = lambda k: {
        "alma:chat:123": b"999",
        "alma:session:last_activity:tg_123": recent_ts,
    }.get(k)
    redis.exists.return_value = False
    await send_proactive("breakfast", redis)
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_skips_crisis_at_boundary():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:456"]
    redis.get.side_effect = lambda k: {
        "alma:chat:456": b"777",
        "alma:crisis:last:tg_456": b"0.61",
    }.get(k)
    redis.exists.return_value = False
    await send_proactive("lunch", redis)
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_allows_low_crisis():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:456"]
    redis.get.side_effect = lambda k: {
        "alma:chat:456": b"777",
        "alma:crisis:last:tg_456": b"0.3",
    }.get(k)
    redis.exists.return_value = False

    with patch("app.scheduler.proactive.httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=mock_resp)))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        await send_proactive("lunch", redis)

    redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_send_proactive_skips_no_chat_id():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:789"]
    redis.get.return_value = None
    redis.exists.return_value = False
    await send_proactive("dinner", redis)
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_handles_telegram_error():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    redis.get.side_effect = lambda k: b"999" if k == "alma:chat:123" else None
    redis.exists.return_value = False

    with patch("app.scheduler.proactive.httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=mock_resp)))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        await send_proactive("breakfast", redis)

    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_handles_network_exception():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:123"]
    redis.get.side_effect = lambda k: b"999" if k == "alma:chat:123" else None
    redis.exists.return_value = False

    with patch("app.scheduler.proactive.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(
            return_value=MagicMock(post=AsyncMock(side_effect=httpx.ConnectError("timeout")))
        )
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        await send_proactive("breakfast", redis)

    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_send_proactive_multiple_users():
    redis = AsyncMock()
    redis.keys.return_value = [b"alma:chat:111", b"alma:chat:222"]
    redis.get.side_effect = lambda k: {
        "alma:chat:111": b"1001",
        "alma:chat:222": b"1002",
    }.get(k)
    redis.exists.return_value = False

    with patch("app.scheduler.proactive.httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=mock_resp)))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        await send_proactive("breakfast", redis)

    assert redis.set.call_count == 2


def test_create_scheduler_warns_on_placeholder_token(caplog):
    import logging
    with patch("app.scheduler.proactive.BOT_TOKEN", "placeholder_fill_me"):
        with caplog.at_level(logging.WARNING):
            redis = AsyncMock()
            create_scheduler(redis)
        assert "TELEGRAM_BOT_TOKEN" in caplog.text


def test_create_scheduler_no_warning_with_real_token(caplog):
    import logging
    with patch("app.scheduler.proactive.BOT_TOKEN", "123456:ABC-DEF"):
        with caplog.at_level(logging.WARNING):
            redis = AsyncMock()
            create_scheduler(redis)
        assert "TELEGRAM_BOT_TOKEN" not in caplog.text


def test_slots_have_correct_messages():
    assert "desayunaste" in SLOTS["breakfast"]["msg"]
    assert "almorzaste" in SLOTS["lunch"]["msg"]
    assert "cenaste" in SLOTS["dinner"]["msg"]


def test_slots_have_correct_hours():
    assert SLOTS["breakfast"]["hour"] == 8
    assert SLOTS["lunch"]["hour"] == 13
    assert SLOTS["dinner"]["hour"] == 19


def test_slots_all_at_minute_30():
    for slot in SLOTS.values():
        assert slot["minute"] == 30


def test_all_slot_names():
    assert set(SLOTS.keys()) == {"breakfast", "lunch", "dinner"}
