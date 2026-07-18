"""WS-H.4 — health (liveness) and ready (deep dep check) endpoints.

/health: shallow process-alive probe. Used by Cloud Run liveness.
/ready: deep dependency check. Hybrid criticality:
    - LLM unreachable → HTTP 503 (Cloud Run pulls the revision)
    - Redis or MCP down → HTTP 200 + status 'degraded' (graceful degrade)
"""
import asyncio
import logging
import os
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

health_router = APIRouter()
ready_router = APIRouter()

REDIS_TIMEOUT_S = 0.5
MCP_TIMEOUT_S = 1.0


@health_router.get("/health")
async def health(request: Request):
    info = getattr(request.app.state, "llm_info", {}) or {}
    return {
        "status": "ok",
        "provider": info.get("provider", "unknown"),
        "model": info.get("model"),
        "preferred": info.get("preferred", "claude-opus-4-7"),
        "fallback_reason": info.get("fallback_reason"),
    }


async def _probe_redis(redis_client) -> tuple[bool, int, "str | None"]:
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(redis_client.ping(), timeout=REDIS_TIMEOUT_S)
        return True, int((time.monotonic() - t0) * 1000), None
    except Exception as exc:
        return (
            False,
            int((time.monotonic() - t0) * 1000),
            f"{type(exc).__name__}: {str(exc)[:140]}",
        )


async def _probe_mcp() -> tuple[bool, int, "str | None"]:
    """HTTP GET on MCP /ready. Cheap because MCP /ready itself is in-process."""
    t0 = time.monotonic()
    mcp_url = os.environ.get("MCP_URL", "http://mcp:8001/mcp")
    base = mcp_url.rstrip("/").removesuffix("/mcp")
    url = f"{base}/ready"
    try:
        async with httpx.AsyncClient(timeout=MCP_TIMEOUT_S) as client:
            resp = await client.get(url)
        ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200 and resp.json().get("status") in ("ok", "degraded"):
            return True, ms, None
        return False, ms, f"mcp /ready returned {resp.status_code}"
    except Exception as exc:
        return (
            False,
            int((time.monotonic() - t0) * 1000),
            f"{type(exc).__name__}: {str(exc)[:140]}",
        )


def _read_llm_health(state) -> dict:
    snap = getattr(state, "llm_health", None)
    if snap is None:
        return {
            "ok": False,
            "error": "not yet probed",
            "stale_s": 0,
            "provider": None,
            "model": None,
            "latency_ms": 0,
        }
    out: dict = {
        "ok": bool(snap.get("ok")),
        "provider": snap.get("provider"),
        "model": snap.get("model"),
        "stale_s": int(time.time() - snap.get("last_probed_at", time.time())),
        "latency_ms": 0,
    }
    if not out["ok"] and snap.get("error"):
        out["error"] = snap["error"]
    return out


@ready_router.get("/ready")
async def ready(request: Request):
    state = request.app.state
    llm_check = _read_llm_health(state)

    redis_client = getattr(state, "scheduler_redis", None)

    async def _do_redis():
        if redis_client is None:
            return {"ok": False, "latency_ms": 0,
                    "error": "redis client not initialized"}
        ok, ms, err = await _probe_redis(redis_client)
        check: dict = {"ok": ok, "latency_ms": ms}
        if err:
            check["error"] = err
        return check

    async def _do_mcp():
        ok, ms, err = await _probe_mcp()
        check: dict = {"ok": ok, "latency_ms": ms}
        if err:
            check["error"] = err
        return check

    redis_check, mcp_check = await asyncio.gather(_do_redis(), _do_mcp())

    checks = {"llm": llm_check, "redis": redis_check, "mcp": mcp_check}

    if not llm_check["ok"]:
        return JSONResponse({"status": "down", "checks": checks}, status_code=503)
    if not (redis_check["ok"] and mcp_check["ok"]):
        return JSONResponse({"status": "degraded", "checks": checks}, status_code=200)
    return JSONResponse({"status": "ok", "checks": checks}, status_code=200)
