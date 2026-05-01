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
    """Returns the current daily counter for the given user_id so the
    frontend can render a "X of N messages today" indicator without
    incrementing the counter."""
    redis = request.app.state.scheduler_redis
    key = f"rate:chat:user:{user_id}:{today_utc()}"
    used = await peek_rate_limit(redis, key)
    return {
        "user_id": user_id,
        "used": used,
        "limit": CHAT_USER_DAILY_LIMIT,
        "remaining": max(0, CHAT_USER_DAILY_LIMIT - used),
    }
