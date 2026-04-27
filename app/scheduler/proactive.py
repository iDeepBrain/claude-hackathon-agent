from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import httpx
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

LIMA_TZ = os.getenv("PROACTIVE_TZ", "America/Lima")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
SILENCE_WINDOW_H = int(os.getenv("PROACTIVE_SILENCE_WINDOW_H", "2"))

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


def create_scheduler(redis_client) -> AsyncIOScheduler:
    if not BOT_TOKEN or BOT_TOKEN.startswith("placeholder"):
        logger.warning("TELEGRAM_BOT_TOKEN not configured — proactive messages will fail")
    scheduler = AsyncIOScheduler()
    for name, cfg in SLOTS.items():
        scheduler.add_job(
            send_proactive,
            CronTrigger(hour=cfg["hour"], minute=cfg["minute"], timezone=LIMA_TZ),
            args=[name, redis_client],
            id=f"proactive_{name}",
            replace_existing=True,
        )
    logger.info(f"Scheduler created with {len(SLOTS)} proactive jobs")
    return scheduler
