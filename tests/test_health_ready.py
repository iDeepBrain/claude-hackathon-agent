"""WS-H.4 — health/ready router tests."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def _make_app(llm_health: dict | None = None, llm_info: dict | None = None) -> FastAPI:
    """Minimal app with health router mounted, no lifespan."""
    from app.api.health import health_router, ready_router
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ready_router, prefix="/api/v1")
    app.state.llm_info = llm_info or {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "preferred": "claude-opus-4-7",
        "fallback_reason": None,
    }
    if llm_health is not None:
        app.state.llm_health = llm_health
    return app


@pytest.mark.asyncio
async def test_health_shallow_returns_200_regardless_of_deps():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-opus-4-7"
    assert body["preferred"] == "claude-opus-4-7"
    assert body["fallback_reason"] is None
