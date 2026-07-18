"""Telegram link-token endpoint — generates a one-time token the
frontend embeds in the bot's deep-link CTA.

The flow (WS-D.4 minimal):

  1. Signed-in web user picks "Telegram" as proactive channel and saves.
  2. Frontend calls POST /api/v1/users/telegram-link/token with their
     google_<sub> user_id; backend generates a random 16-char hex token,
     stores ``alma:tg-link:<token>`` → user_id in Redis with a 10-min
     TTL, and returns ``{token, deep_link}``.
  3. Frontend opens the deep-link, e.g.
     ``https://t.me/AlmaHackathonBot?start=alma_<token>``.
  4. User clicks. Telegram opens the bot. Bot's /start handler reads
     the start param, looks up the token in Redis, retrieves the
     google_<sub>, and persists the chat_id → user_id mapping
     (``alma:tg-chat-for:<user_id>`` → chat_id) so the proactive
     scheduler can call sendMessage(chat_id, …) when it fires.

Security: tokens are 16 hex chars (64 bits of entropy) — collision
risk is negligible at the scale of "tens of users per minute."
The token is single-use (deleted after the bot consumes it). TTL
shrinks the window in which a leaked token is exploitable.
"""
from __future__ import annotations

import logging
import os
import secrets

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["telegram_link"])


_LINK_TTL_SEC = 600  # 10 minutes
_TOKEN_BYTES = 8  # 8 bytes → 16 hex chars


class TelegramLinkTokenRequest(BaseModel):
    user_id: str = Field(..., description="Canonical google_<sub> from /auth/google")


class TelegramLinkTokenResponse(BaseModel):
    token: str
    deep_link: str


def _build_deep_link(bot_username: str, token: str) -> str:
    # Telegram requires the start parameter to be a single token
    # (no spaces, [a-zA-Z0-9_-]). We prefix with "alma_" so the bot
    # handler can distinguish our tokens from any other start params.
    return f"https://t.me/{bot_username}?start=alma_{token}"


async def _get_redis(request: Request) -> aioredis.Redis:
    """Reuse the agent's shared scheduler_redis when available; otherwise
    create a fresh client. Same pattern as cron.py."""
    redis_client = getattr(request.app.state, "scheduler_redis", None)
    if redis_client is None:
        redis_url = os.environ["REDIS_URL"]
        redis_client = aioredis.from_url(redis_url)
        request.app.state.scheduler_redis = redis_client
    return redis_client


@router.post("/users/telegram-link/token", response_model=TelegramLinkTokenResponse)
async def issue_telegram_link_token(
    req: TelegramLinkTokenRequest, request: Request
) -> TelegramLinkTokenResponse:
    """Issue a one-time token the user clicks to link their Telegram
    chat to their authenticated identity. Token expires after 10 min."""
    bot_username = os.getenv("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@")
    if not bot_username:
        raise HTTPException(
            status_code=503,
            detail="Telegram link disabled (TELEGRAM_BOT_USERNAME env unset)",
        )

    if not req.user_id.startswith("google_") and not req.user_id.startswith("tg_"):
        # Defense-in-depth: anonymous UUIDs cannot link a bot to themselves.
        raise HTTPException(status_code=400, detail="Link requires authenticated identity")

    token = secrets.token_hex(_TOKEN_BYTES)
    redis_client = await _get_redis(request)
    await redis_client.set(f"alma:tg-link:{token}", req.user_id, ex=_LINK_TTL_SEC)

    logger.info("Issued telegram-link token for user_id=%s", req.user_id)
    return TelegramLinkTokenResponse(
        token=token,
        deep_link=_build_deep_link(bot_username, token),
    )
