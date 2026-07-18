"""Daily rate limiting via Redis INCR + EXPIRE.

Used by the public chat endpoint and (via the Telegram bot) the
``handlers/text.py`` flow to keep token spend bounded when the demo URL
is shared in public.

Keys follow the convention ``rate:<scope>:<id>:<YYYYMMDD>`` and expire
after 24h, so the counter resets at UTC midnight.
"""
from __future__ import annotations

from datetime import datetime, timezone


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


async def check_rate_limit(redis, key: str, limit: int, window_seconds: int = 86400) -> tuple[bool, int]:
    """Increment ``key`` and return ``(allowed, current_count)``.

    The first INCR sets the counter to 1; we then arm an EXPIRE so the
    key vanishes after ``window_seconds``. Subsequent INCRs piggy-back on
    the existing TTL.
    """
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_seconds)
    return (count <= limit, int(count))


async def peek_rate_limit(redis, key: str) -> int:
    """Read the current counter value without incrementing. Returns 0 if absent."""
    raw = await redis.get(key)
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0
