"""Cloud Scheduler endpoints for proactive check-ins.

Cloud Scheduler hits these on a cron schedule (08:30, 13:30, 19:30 Lima time)
instead of running APScheduler in-process — which avoids duplicate sends when
Cloud Run autoscales to multiple replicas.

Each endpoint is protected by the X-Cloud-Scheduler-Token header (verified
against the CRON_TOKEN env var). Idempotence is preserved via the same Redis
slot key the APScheduler path uses.
"""
from __future__ import annotations

import logging
import os

import redis.asyncio as aioredis
from fastapi import APIRouter, Header, HTTPException, Path, Request

from app.scheduler.proactive import SLOTS, send_proactive

logger = logging.getLogger(__name__)
router = APIRouter()

EXPECTED_TOKEN = os.getenv("CRON_TOKEN", "")


def _verify_token(token: str | None) -> None:
    if not EXPECTED_TOKEN:
        logger.warning("CRON_TOKEN not configured — cron endpoint accepts unauthenticated calls")
        return
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid X-Cloud-Scheduler-Token")


@router.post("/cron/proactive/{slot}")
async def proactive_cron(
    request: Request,
    slot: str = Path(..., pattern="^(breakfast|lunch|dinner)$"),
    x_cloud_scheduler_token: str | None = Header(default=None, alias="X-Cloud-Scheduler-Token"),
):
    _verify_token(x_cloud_scheduler_token)

    if slot not in SLOTS:
        raise HTTPException(status_code=400, detail=f"Unknown slot '{slot}'")

    redis_client = getattr(request.app.state, "scheduler_redis", None)
    if redis_client is None:
        redis_url = os.environ["REDIS_URL"]
        redis_client = aioredis.from_url(redis_url)
        request.app.state.scheduler_redis = redis_client

    await send_proactive(slot, redis_client)
    logger.info("cron/proactive/%s executed", slot)
    return {"status": "ok", "slot": slot}
