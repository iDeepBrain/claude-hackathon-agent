"""WS-D.3 — Web Push subscription management.

Two endpoints, both gated by Google id_token verification (re-using the
verify helper in app/api/auth.py):

  POST   /api/v1/users/push-subscription  → store subscription JSONB
  DELETE /api/v1/users/push-subscription  → wipe subscription (idempotent)

The subscription itself is the standard PushSubscription shape returned
by the browser API:

  { "endpoint": "https://...", "keys": {"p256dh": "...", "auth": "..."} }

Stored as-is in alma_users.push_subscription (JSONB). The scheduler
reads it back when sending pushes.

Privacy note: storing the endpoint URL is necessary — it IS the address
the push service uses. We never log the full URL (only the host and a
truncated path) to keep audit logs PII-safe.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.api.auth import _verify_google_token  # internal helper, intentional
from app.push.web_push import is_configured as _push_configured

logger = logging.getLogger(__name__)
router = APIRouter(tags=["push"])


class PushKeys(BaseModel):
    p256dh: str = Field(min_length=8, max_length=200)
    auth: str = Field(min_length=8, max_length=200)


class PushSubscription(BaseModel):
    endpoint: str = Field(min_length=8, max_length=2048)
    keys: PushKeys


class SaveSubscriptionRequest(BaseModel):
    id_token: str
    subscription: PushSubscription


class DeleteSubscriptionRequest(BaseModel):
    id_token: str


def _redact_endpoint(url: str) -> str:
    """Return host + first 12 chars of path for logs. Never the full URL."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return f"{p.hostname}{p.path[:12] + '…' if len(p.path) > 12 else p.path}"
    except Exception:
        return "<unparseable>"


async def _update_user_push(user_id: str, payload: dict | None) -> bool:
    """Set or clear alma_users.push_subscription. Returns True if a row was
    updated, False if no such user_id exists.

    Direct DB access matches the existing pattern in scripts/reset_local.py;
    runtime code that needs heavy memory operations goes through MCP, but
    this is a single column update — round-tripping through MCP would add
    complexity for no gain.
    """
    import json as _json
    import os

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as session:
            async with session.begin():
                if payload is None:
                    res = await session.execute(
                        text(
                            "UPDATE alma_users SET push_subscription=NULL, "
                            "push_disabled_at=NULL WHERE user_id=:uid"
                        ),
                        {"uid": user_id},
                    )
                else:
                    res = await session.execute(
                        text(
                            "UPDATE alma_users SET push_subscription=CAST(:sub AS jsonb), "
                            "push_disabled_at=NULL WHERE user_id=:uid"
                        ),
                        {"uid": user_id, "sub": _json.dumps(payload)},
                    )
                return (res.rowcount or 0) > 0
    finally:
        await engine.dispose()


@router.post("/users/push-subscription")
async def save_push_subscription(req: SaveSubscriptionRequest) -> dict:
    if not _push_configured():
        raise HTTPException(
            status_code=503,
            detail="Web Push disabled (VAPID env not configured)",
        )

    info = _verify_google_token(req.id_token)
    user_id = f"google_{info['sub']}"

    sub_dict = req.subscription.model_dump()
    updated = await _update_user_push(user_id, sub_dict)
    if not updated:
        # The /api/v1/auth/google call should have created the row earlier
        # via /api/v1/memory upsert. If not, something is wrong upstream —
        # we don't silently insert (could mask a bug).
        logger.warning("push-subscription POST for missing user_id=%s", user_id)
        raise HTTPException(
            status_code=404,
            detail="user not found — log in and create a profile first",
        )

    logger.info(
        "Push subscription saved for user=%s endpoint=%s",
        user_id, _redact_endpoint(sub_dict["endpoint"]),
    )
    return {"ok": True}


@router.delete("/users/push-subscription")
async def delete_push_subscription(req: DeleteSubscriptionRequest) -> dict:
    info = _verify_google_token(req.id_token)
    user_id = f"google_{info['sub']}"

    # Idempotent — fine if subscription was already null.
    await _update_user_push(user_id, None)
    logger.info("Push subscription cleared for user=%s", user_id)
    return {"ok": True}
