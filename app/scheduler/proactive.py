from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import httpx
import os
import logging
from datetime import datetime, timedelta, timezone

from app.push.web_push import build_payload, is_configured, send_push  # noqa: F401

logger = logging.getLogger(__name__)

LIMA_TZ = os.getenv("PROACTIVE_TZ", "America/Lima")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SILENCE_WINDOW_H = int(os.getenv("PROACTIVE_SILENCE_WINDOW_H", "2"))

# WS-D.3 — auto-pause threshold (days of inactivity before we stop pushing).
PUSH_AUTO_PAUSE_DAYS = int(os.getenv("PROACTIVE_PUSH_AUTOPAUSE_DAYS", "7"))

SLOTS = {
    "breakfast": {
        "hour": int(os.getenv("PROACTIVE_BREAKFAST_H", "8")),
        "minute": 30,
        "msg": "¿Ya desayunaste? ☀️ Un buen comienzo importa",
    },
    "lunch": {
        "hour": int(os.getenv("PROACTIVE_LUNCH_H", "13")),
        "minute": 30,
        "msg": "¿Ya almorzaste? 🌞 ¿Cómo va tu día?",
    },
    "dinner": {
        "hour": int(os.getenv("PROACTIVE_DINNER_H", "19")),
        "minute": 30,
        "msg": "¿Ya cenaste? 🌙 ¿Hiciste algo de movimiento hoy?",
    },
}


async def send_proactive(slot_name: str, redis_client):
    today = datetime.now().strftime("%Y-%m-%d")
    slot_cfg = SLOTS[slot_name]

    keys = [k.decode() if isinstance(k, bytes) else k for k in await redis_client.keys("alma:chat:*")]

    for key in keys:
        tg_user_id = key.split(":")[-1]
        user_id = f"tg_{tg_user_id}"
        raw_chat_id = await redis_client.get(key)
        if not raw_chat_id:
            continue
        chat_id = raw_chat_id.decode() if isinstance(raw_chat_id, bytes) else raw_chat_id

        slot_key = f"alma:proactive:slot:{user_id}:{today}:{slot_name}"
        if await redis_client.exists(slot_key):
            continue

        last_activity = await redis_client.get(f"alma:session:last_activity:{user_id}")
        if last_activity:
            ts = float(last_activity.decode() if isinstance(last_activity, bytes) else last_activity)
            if datetime.now().timestamp() - ts < SILENCE_WINDOW_H * 3600:
                continue

        crisis = await redis_client.get(f"alma:crisis:last:{user_id}")
        if crisis:
            score = float(crisis.decode() if isinstance(crisis, bytes) else crisis)
            if score > 0.6:
                continue

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={"chat_id": int(chat_id), "text": slot_cfg["msg"]},
                )
                if resp.status_code == 200:
                    await redis_client.set(slot_key, "1", ex=86400)
                    logger.info(f"Proactive {slot_name} sent to {user_id}")
                else:
                    logger.warning(f"Telegram API error for {user_id}: {resp.status_code}")
        except Exception as e:
            logger.error(f"Proactive send failed for {user_id}: {e}")


async def _fetch_active_push_users() -> list[dict]:
    """WS-D.3 — Return rows with push subscriptions worth touching this tick.

    Each dict has: user_id, push_subscription, last_seen_at (datetime).
    Filters at SQL level: subscription not null, not auto-paused.
    """
    import os

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return []

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            res = await session.execute(text(
                "SELECT user_id, push_subscription, last_seen_at "
                "FROM alma_users "
                "WHERE push_subscription IS NOT NULL "
                "  AND push_disabled_at IS NULL"
            ))
            return [
                {
                    "user_id": row[0],
                    "push_subscription": row[1],
                    "last_seen_at": row[2],
                }
                for row in res.all()
            ]
    finally:
        await engine.dispose()


async def _mark_push_disabled(user_id: str, reason: str) -> None:
    """Set push_disabled_at=now on the user row. Idempotent."""
    import os

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE alma_users SET push_disabled_at=now() WHERE user_id=:uid"),
                    {"uid": user_id},
                )
        logger.info("Push auto-paused user_id=%s reason=%s", user_id, reason)
    finally:
        await engine.dispose()


async def _clear_push_subscription(user_id: str) -> None:
    """Wipe the subscription after a 410 Gone — the endpoint is dead."""
    import os

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE alma_users SET push_subscription=NULL WHERE user_id=:uid"),
                    {"uid": user_id},
                )
        logger.info("Push subscription cleared (410) for user_id=%s", user_id)
    finally:
        await engine.dispose()


async def send_proactive_push(slot_name: str, redis_client):
    """WS-D.3 — Web push variant of send_proactive.

    Fires alongside the Telegram path on every slot tick. Uses Redis to
    enforce the same gates (silence window, crisis, slot-already-sent),
    plus an extra DB-side gate for subscription state and auto-pause.
    """
    if not is_configured():
        return  # VAPID env unset — push channel is dark by design

    today = datetime.now().strftime("%Y-%m-%d")
    slot_cfg = SLOTS[slot_name]

    users = await _fetch_active_push_users()
    if not users:
        return

    pause_threshold = datetime.now(timezone.utc) - timedelta(days=PUSH_AUTO_PAUSE_DAYS)

    for user in users:
        user_id = user["user_id"]

        # Auto-pause for ghost users.
        last_seen = user["last_seen_at"]
        if last_seen and last_seen < pause_threshold:
            await _mark_push_disabled(user_id, f"inactive >{PUSH_AUTO_PAUSE_DAYS}d")
            continue

        # Slot already sent today?
        slot_key = f"alma:proactive:slot:{user_id}:{today}:{slot_name}"
        if await redis_client.exists(slot_key):
            continue

        # Silence window (recent activity)?
        last_activity = await redis_client.get(f"alma:session:last_activity:{user_id}")
        if last_activity:
            ts = float(last_activity.decode() if isinstance(last_activity, bytes) else last_activity)
            if datetime.now().timestamp() - ts < SILENCE_WINDOW_H * 3600:
                continue

        # Crisis gate.
        crisis = await redis_client.get(f"alma:crisis:last:{user_id}")
        if crisis:
            score = float(crisis.decode() if isinstance(crisis, bytes) else crisis)
            if score > 0.6:
                continue

        # Send.
        payload = build_payload(slot_name, "Hola, soy Alma", slot_cfg["msg"])
        try:
            outcome = await send_push(user["push_subscription"], payload)
        except Exception as exc:
            logger.error("Push send raised for %s: %s", user_id, exc)
            continue

        if outcome == "sent":
            await redis_client.set(slot_key, "1", ex=86400)
            logger.info("Push %s sent to %s", slot_name, user_id)
        elif outcome == "expired":
            await _clear_push_subscription(user_id)
        # "failed" → log only; retry on next slot


def create_scheduler(redis_client) -> AsyncIOScheduler:
    if not BOT_TOKEN or BOT_TOKEN.startswith("placeholder"):
        logger.warning("TELEGRAM_BOT_TOKEN not configured — Telegram proactive disabled")
    scheduler = AsyncIOScheduler()
    for name, cfg in SLOTS.items():
        scheduler.add_job(
            send_proactive,
            CronTrigger(hour=cfg["hour"], minute=cfg["minute"], timezone=LIMA_TZ),
            args=[name, redis_client],
            id=f"proactive_{name}",
            replace_existing=True,
        )
        # WS-D.3 — second job per slot for the web push channel.
        scheduler.add_job(
            send_proactive_push,
            CronTrigger(hour=cfg["hour"], minute=cfg["minute"], timezone=LIMA_TZ),
            args=[name, redis_client],
            id=f"proactive_push_{name}",
            replace_existing=True,
        )
    logger.info(f"Scheduler created with {len(SLOTS) * 2} proactive jobs (telegram + push)")
    return scheduler
