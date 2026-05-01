"""Chat endpoint — SSE bridge between AlmaChain and the browser.

Emits two kinds of SSE events on the same stream:

- **Unnamed events** (``data: <token>``): the response token stream. The
  existing browser client consumes these in ``EventSource.onmessage`` and
  appends them to the visible chat. This contract is preserved for backward
  compatibility — adding the trace panel must not break the chat.
- **Named events** (``event: <type>\\ndata: <json>``): pipeline observability
  for the agent-trace panel. New clients add ``addEventListener('agent_start',
  ...)``, ``addEventListener('memory_retrieved', ...)``, etc. Old clients that
  only listen to ``onmessage`` ignore them silently.

The payload of named events never includes raw memory chunks or the user's
message text — only metadata (counts, scores, layers, latency, model name).
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.utils.rate_limit import check_rate_limit, peek_rate_limit, today_utc

router = APIRouter()

# Public-demo limits. Override at deploy time without touching code.
CHAT_USER_DAILY_LIMIT = int(os.getenv("CHAT_USER_DAILY_LIMIT", "30"))
CHAT_IP_DAILY_LIMIT = int(os.getenv("CHAT_IP_DAILY_LIMIT", "100"))
# Max raw chars per message — bounds the token budget per request and
# blocks unbounded-consumption DoS attempts (OWASP LLM10). Default 500
# is generous for a single emotional-support turn (~80–100 words) while
# keeping per-request cost predictable. Override with the env var when
# legitimate long-form content is expected.
CHAT_MAX_MESSAGE_CHARS = int(os.getenv("CHAT_MAX_MESSAGE_CHARS", "500"))


class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: str | None = None
    language: str = "es"  # es | en


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Honours X-Forwarded-For / X-Real-IP when set
    by the reverse proxy (nginx / Cloudflare). Falls back to the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        # Left-most entry is the original client; everything else is proxies.
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    if len(req.message or "") > CHAT_MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "message_too_long",
                "message": f"Message exceeds {CHAT_MAX_MESSAGE_CHARS} characters. Send something shorter 💛",
                "limit": CHAT_MAX_MESSAGE_CHARS,
            },
        )

    redis = request.app.state.scheduler_redis
    today = today_utc()

    user_key = f"rate:chat:user:{req.user_id}:{today}"
    ip_key = f"rate:chat:ip:{_client_ip(request)}:{today}"

    user_ok, _ = await check_rate_limit(redis, user_key, CHAT_USER_DAILY_LIMIT)
    ip_ok, _ = await check_rate_limit(redis, ip_key, CHAT_IP_DAILY_LIMIT)

    if not (user_ok and ip_ok):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "message": "You've reached today's message limit on the public demo. Come back tomorrow 💛",
                "limit": CHAT_USER_DAILY_LIMIT,
            },
        )

    alma_chain = request.app.state.alma_chain

    async def generate():
        async for event in alma_chain.stream_events(
            req.user_id, req.message, req.image_base64, req.language
        ):
            event_type = event.get("type")
            if event_type == "response_chunk":
                # Backward-compatible token stream — onmessage handler appends these
                yield {"data": event.get("content", "")}
            else:
                # Named event for the agent trace panel — addEventListener consumes these
                payload = {k: v for k, v in event.items() if k != "type"}
                yield {"event": event_type, "data": json.dumps(payload, default=str)}

    return EventSourceResponse(generate())


@router.get("/chat/usage")
async def chat_usage(user_id: str, request: Request):
    """Returns daily counters for the badge.

    Two counters are tracked server-side: per user_id (UUID) and per
    client IP. The badge needs a single "X / limit" pair to render —
    we expose both raw counters and an "effective" projection that
    reflects whichever cap the caller is closest to.

    A user who clears cookies resets their user_id counter to 0, but
    the IP counter keeps climbing. The badge using ``used``/``limit``
    will then surface the IP-side pressure proportionally so the user
    sees realistic remaining capacity, not a misleading 0/30.
    """
    redis = request.app.state.scheduler_redis
    today = today_utc()
    user_key = f"rate:chat:user:{user_id}:{today}"
    ip_key = f"rate:chat:ip:{_client_ip(request)}:{today}"
    user_used = await peek_rate_limit(redis, user_key)
    ip_used = await peek_rate_limit(redis, ip_key)

    # Effective = whichever counter sits closer to its cap, projected
    # back into the user-counter scale so the badge math stays "X / 30".
    user_pct = (user_used / CHAT_USER_DAILY_LIMIT) if CHAT_USER_DAILY_LIMIT else 0.0
    ip_pct = (ip_used / CHAT_IP_DAILY_LIMIT) if CHAT_IP_DAILY_LIMIT else 0.0
    effective_pct = max(user_pct, ip_pct)
    effective_used = min(CHAT_USER_DAILY_LIMIT, round(effective_pct * CHAT_USER_DAILY_LIMIT))

    return {
        "user_id": user_id,
        # Primary fields the badge reads — kept stable for backward compat.
        "used": effective_used,
        "limit": CHAT_USER_DAILY_LIMIT,
        "remaining": max(0, CHAT_USER_DAILY_LIMIT - effective_used),
        # Raw counters for observability + future UI variants.
        "user_used": user_used,
        "user_limit": CHAT_USER_DAILY_LIMIT,
        "ip_used": ip_used,
        "ip_limit": CHAT_IP_DAILY_LIMIT,
    }
