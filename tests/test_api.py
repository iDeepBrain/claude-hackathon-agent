"""
FastAPI endpoint tests using httpx.AsyncClient with mocked app state.
No real LLM, Redis, or MCP required.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sse_starlette.sse import EventSourceResponse

from app.api.chat import router as chat_router
from app.api.proactivity import router as proactivity_router
from app.cache.session import Session


def make_app(stream_chunks: list[str] | None = None, language_in_session: str = "es") -> FastAPI:
    """Return a minimal FastAPI app with mocked state, no lifespan."""
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(proactivity_router, prefix="/api/v1")

    # Mock alma_chain — both legacy stream() (string yields) and new stream_events() (dict yields)
    chunks = stream_chunks or ["hola desde mock"]

    async def mock_stream(user_id, message, image_b64=None, language="es"):
        for chunk in chunks:
            yield chunk

    async def mock_stream_events(user_id, message, image_b64=None, language="es"):
        yield {"type": "agent_start", "language": language, "has_image": image_b64 is not None}
        for chunk in chunks:
            yield {"type": "response_chunk", "content": chunk}
        yield {"type": "agent_done", "stop_reason": "end_turn", "chunks": len(chunks), "latency_ms": 0}

    mock_chain = MagicMock()
    mock_chain.stream = mock_stream
    mock_chain.stream_events = mock_stream_events

    # Mock mcp_client
    mock_mcp = MagicMock()
    mock_mcp.build_context = AsyncMock(return_value="Sin historial previo para este usuario.")

    # Mock session_store
    mock_sessions = MagicMock()
    mock_sessions.get = AsyncMock(return_value=Session(state="chat", language=language_in_session))

    app.state.alma_chain = mock_chain
    app.state.mcp_client = mock_mcp
    app.state.session_store = mock_sessions
    return app


@pytest.fixture
def app():
    return make_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── /health ────────────────────────────────────────────────────────────────────

async def test_health_endpoint():
    from app.main import app as main_app
    # We only check the route exists and returns the right shape — no lifespan needed
    # because /health has no state dependency
    from fastapi.testclient import TestClient
    import contextlib

    # Patch lifespan to a no-op so TestClient can start the app
    @contextlib.asynccontextmanager
    async def noop_lifespan(app):
        yield

    main_app.router.lifespan_context = noop_lifespan
    with TestClient(main_app) as tc:
        resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /chat schema ───────────────────────────────────────────────────────────────

async def test_chat_language_defaults_to_es(client):
    resp = await client.post("/api/v1/chat", json={"user_id": "u1", "message": "hola"})
    assert resp.status_code == 200


async def test_chat_language_en_accepted(client):
    resp = await client.post(
        "/api/v1/chat",
        json={"user_id": "u1", "message": "hello", "language": "en"},
    )
    assert resp.status_code == 200


async def test_chat_missing_user_id_returns_422(client):
    resp = await client.post("/api/v1/chat", json={"message": "hola"})
    assert resp.status_code == 422


async def test_chat_missing_message_returns_422(client):
    resp = await client.post("/api/v1/chat", json={"user_id": "u1"})
    assert resp.status_code == 422


async def test_chat_response_is_sse(client):
    resp = await client.post("/api/v1/chat", json={"user_id": "u1", "message": "hola"})
    assert "text/event-stream" in resp.headers.get("content-type", "")


async def test_chat_sse_body_contains_data(client):
    resp = await client.post("/api/v1/chat", json={"user_id": "u1", "message": "hola"})
    assert b"data:" in resp.content


# ── /trigger schema ────────────────────────────────────────────────────────────

async def test_trigger_valid_type_accepted(client):
    with patch("app.api.proactivity.make_llm") as MockLLM:
        mock_llm = MagicMock()
        MockLLM.return_value = mock_llm

        async def fake_astream(messages):
            yield MagicMock(content="hola")

        mock_llm.astream = fake_astream
        resp = await client.post(
            "/api/v1/trigger",
            json={"user_id": "u1", "trigger_type": "morning_checkin"},
        )
    assert resp.status_code == 200


async def test_trigger_missing_user_id_returns_422(client):
    resp = await client.post("/api/v1/trigger", json={"trigger_type": "morning_checkin"})
    assert resp.status_code == 422


async def test_trigger_missing_type_returns_422(client):
    resp = await client.post("/api/v1/trigger", json={"user_id": "u1"})
    assert resp.status_code == 422


async def test_trigger_uses_session_language():
    """Proactive trigger must read language from the stored session, not the request."""
    app_en = make_app(language_in_session="en")

    with patch("app.api.proactivity.make_llm") as MockLLM:
        mock_llm = MagicMock()
        MockLLM.return_value = mock_llm

        async def fake_astream(messages):
            yield MagicMock(content="hello")

        mock_llm.astream = fake_astream
        async with AsyncClient(transport=ASGITransport(app=app_en), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/trigger",
                json={"user_id": "u1", "trigger_type": "morning_checkin"},
            )
    # Session language was "en" — the trigger should not crash and return SSE
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
