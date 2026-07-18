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


# ── LLM refresher tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresher_writes_initial_state():
    """probe_once snapshots probe result into a dict."""
    from app.safety.llm_refresher import probe_once

    async def fake_anthropic(model):
        return True, None

    async def fake_gemini(model):
        return False, "GOOGLE_API_KEY not set"

    with patch("app.safety.llm_refresher._ping_anthropic", new=fake_anthropic), \
         patch("app.safety.llm_refresher._ping_gemini", new=fake_gemini), \
         patch("app.safety.llm_refresher.PROVIDER_CHAIN", new=[
             ("anthropic", "claude-opus-4-7", 4096),
         ]):
        result = await probe_once()

    assert result["ok"] is True
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-opus-4-7"
    assert result["error"] is None
    assert "last_probed_at" in result


@pytest.mark.asyncio
async def test_refresher_handles_probe_exception():
    from app.safety.llm_refresher import probe_once

    async def boom(model):
        raise RuntimeError("network exploded")

    with patch("app.safety.llm_refresher._ping_anthropic", new=boom), \
         patch("app.safety.llm_refresher._ping_gemini", new=boom), \
         patch("app.safety.llm_refresher.PROVIDER_CHAIN", new=[
             ("anthropic", "claude-opus-4-7", 4096),
             ("gemini", "gemini-2.0-flash", 4096),
         ]):
        result = await probe_once()

    assert result["ok"] is False
    assert result["provider"] is None
    assert result["error"] is not None
    assert "RuntimeError" in result["error"]


@pytest.mark.asyncio
async def test_refresher_loop_writes_state_periodically():
    """Spawn loop, wait one tick, assert it wrote then cancelled cleanly."""
    from app.safety.llm_refresher import refresher_loop
    app = FastAPI()
    app.state.llm_health = {"ok": False, "provider": None, "model": None,
                            "error": "not yet probed", "last_probed_at": 0.0}

    async def quick_probe():
        return {"ok": True, "provider": "anthropic", "model": "claude-opus-4-7",
                "error": None, "last_probed_at": time.time()}

    with patch("app.safety.llm_refresher.probe_once", new=quick_probe):
        task = asyncio.create_task(refresher_loop(app, interval_s=0.05))
        await asyncio.sleep(0.12)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert app.state.llm_health["ok"] is True
    assert app.state.llm_health["provider"] == "anthropic"


# ── /api/v1/ready tests ───────────────────────────────────────────────────────


def _make_app_with_state(llm_health, redis_ping=None):
    from app.api.health import health_router, ready_router
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ready_router, prefix="/api/v1")
    app.state.llm_info = {"provider": "anthropic", "model": "claude-opus-4-7",
                          "preferred": "claude-opus-4-7", "fallback_reason": None}
    app.state.llm_health = llm_health

    redis_mock = MagicMock()
    if redis_ping is None:
        redis_mock.ping = AsyncMock(return_value=True)
    else:
        redis_mock.ping = redis_ping
    app.state.scheduler_redis = redis_mock

    return app


@pytest.mark.asyncio
async def test_ready_all_ok_returns_200_status_ok():
    app = _make_app_with_state(
        llm_health={"ok": True, "provider": "anthropic",
                    "model": "claude-opus-4-7", "error": None,
                    "last_probed_at": time.time() - 5},
    )
    with patch("app.api.health._probe_mcp", AsyncMock(return_value=(True, 47, None))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["llm"]["ok"] is True
    assert body["checks"]["llm"]["provider"] == "anthropic"
    assert body["checks"]["llm"]["stale_s"] >= 4
    assert body["checks"]["redis"]["ok"] is True
    assert body["checks"]["mcp"]["ok"] is True


@pytest.mark.asyncio
async def test_ready_redis_down_returns_200_status_degraded():
    app = _make_app_with_state(
        llm_health={"ok": True, "provider": "anthropic", "model": "claude-opus-4-7",
                    "error": None, "last_probed_at": time.time()},
        redis_ping=AsyncMock(side_effect=asyncio.TimeoutError()),
    )
    with patch("app.api.health._probe_mcp", AsyncMock(return_value=(True, 30, None))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"]["ok"] is False
    assert "TimeoutError" in body["checks"]["redis"]["error"]


@pytest.mark.asyncio
async def test_ready_mcp_down_returns_200_status_degraded():
    app = _make_app_with_state(
        llm_health={"ok": True, "provider": "anthropic", "model": "claude-opus-4-7",
                    "error": None, "last_probed_at": time.time()},
    )
    with patch("app.api.health._probe_mcp",
               AsyncMock(return_value=(False, 1000, "ConnectError: connection refused"))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["checks"]["mcp"]["ok"] is False


@pytest.mark.asyncio
async def test_ready_llm_down_returns_503_status_down():
    app = _make_app_with_state(
        llm_health={"ok": False, "provider": None, "model": None,
                    "error": "all providers exhausted", "last_probed_at": time.time()},
    )
    with patch("app.api.health._probe_mcp", AsyncMock(return_value=(True, 30, None))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "down"
    assert r.json()["checks"]["llm"]["ok"] is False


@pytest.mark.asyncio
async def test_ready_llm_not_probed_yet_returns_503():
    """Cold-start fail-closed: if llm_health is missing, /ready returns 503."""
    from app.api.health import health_router, ready_router
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ready_router, prefix="/api/v1")
    redis_mock = MagicMock()
    redis_mock.ping = AsyncMock(return_value=True)
    app.state.scheduler_redis = redis_mock
    with patch("app.api.health._probe_mcp", AsyncMock(return_value=(True, 30, None))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["llm"]["error"] == "not yet probed"


@pytest.mark.asyncio
async def test_ready_runs_probes_in_parallel():
    async def slow_ping():
        await asyncio.sleep(0.4)
        return True

    async def slow_mcp():
        await asyncio.sleep(0.4)
        return True, 400, None

    redis_mock = MagicMock()
    redis_mock.ping = slow_ping

    from app.api.health import health_router, ready_router
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ready_router, prefix="/api/v1")
    app.state.llm_info = {"provider": "anthropic", "model": "claude-opus-4-7",
                          "preferred": "claude-opus-4-7", "fallback_reason": None}
    app.state.llm_health = {"ok": True, "provider": "anthropic", "model": "claude-opus-4-7",
                            "error": None, "last_probed_at": time.time()}
    app.state.scheduler_redis = redis_mock

    with patch("app.api.health._probe_mcp", new=slow_mcp):
        t0 = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/ready")
        elapsed = time.monotonic() - t0

    assert r.status_code == 200
    assert elapsed < 0.7, f"probes ran sequentially: {elapsed:.2f}s"
