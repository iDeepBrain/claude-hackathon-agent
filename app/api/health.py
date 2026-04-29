"""WS-H.4 — health (liveness) and ready (deep dep check) endpoints.

/health: shallow process-alive probe. Used by Cloud Run liveness.
/ready: deep dependency check. Hybrid criticality:
    - LLM unreachable → HTTP 503 (Cloud Run pulls the revision)
    - Redis or MCP down → HTTP 200 + status 'degraded' (graceful degrade)
"""
from fastapi import APIRouter, Request

health_router = APIRouter()
ready_router = APIRouter()


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
