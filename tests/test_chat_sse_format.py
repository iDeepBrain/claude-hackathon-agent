"""SSE byte-format integration tests for the ``/api/v1/chat`` endpoint.

These tests assert the EXACT bytes the endpoint emits — they are what the
existing test_api.py + test_chain.py suite was missing. The previous suite
mocked ``stream_events`` so it never observed the on-the-wire SSE format
that real consumers (web ``chat.js``, telegram ``agent_client.py``) parse.

Without these tests, a regression in chat.py's named-event formatting would
ship to production unnoticed — exactly what happened in the post-deploy
bug where named-event JSON payloads leaked through old client parsers.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.chat import router as chat_router
from app.cache.session import Session


# ── helpers ──────────────────────────────────────────────────────────────────


def make_app_with_events(events: list[dict]) -> FastAPI:
    """Build a minimal FastAPI app that returns the given event sequence
    when /api/v1/chat is POSTed to."""
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")

    async def mock_stream_events(user_id, message, image_b64=None, language="es"):
        for ev in events:
            yield ev

    mock_chain = MagicMock()
    mock_chain.stream_events = mock_stream_events

    mock_mcp = MagicMock()
    mock_mcp.build_context = AsyncMock(return_value="")

    mock_sessions = MagicMock()
    mock_sessions.get = AsyncMock(return_value=Session(state="chat", language="es"))

    app.state.alma_chain = mock_chain
    app.state.mcp_client = mock_mcp
    app.state.session_store = mock_sessions
    return app


async def post_chat_body(app: FastAPI, message: str = "hi") -> bytes:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/chat", json={"user_id": "u1", "message": message})
        assert resp.status_code == 200
        return resp.content


# ── named events render with `event: <name>\r\n` prefix ─────────────────────


@pytest.mark.asyncio
async def test_named_event_serializes_with_event_prefix():
    """Each non-``response_chunk`` event must produce ``event: <type>\\r\\n``
    so SSE clients can opt-in via ``addEventListener`` and parse separately
    from token chunks."""
    app = make_app_with_events([
        {"type": "agent_start", "language": "es", "has_image": False},
        {"type": "response_chunk", "content": "ok"},
        {"type": "agent_done", "stop_reason": "end_turn", "latency_ms": 12},
    ])
    body = await post_chat_body(app)
    assert b"event: agent_start\r\n" in body
    assert b"event: agent_done\r\n" in body
    # response_chunk gets unnamed `data:` line, NOT `event: response_chunk`
    assert b"event: response_chunk" not in body


@pytest.mark.asyncio
async def test_named_event_payload_is_json_without_type_field():
    """The payload after ``data:`` must be valid JSON containing the metadata
    fields, with the ``type`` field stripped (it's already in ``event:``)."""
    app = make_app_with_events([
        {"type": "memory_retrieved", "count": 3, "layers": ["mood_history"], "top_score": 0.87},
    ])
    body = await post_chat_body(app)
    assert b"event: memory_retrieved\r\n" in body
    # Field types preserved
    assert b'"count": 3' in body
    assert b'"layers": ["mood_history"]' in body
    assert b'"top_score": 0.87' in body
    # type field NOT in payload
    assert b'"type":' not in body


# ── unnamed (token) events MUST NOT have `event:` line ─────────────────────


@pytest.mark.asyncio
async def test_response_chunk_emits_unnamed_data_line_only():
    """Tokens must arrive as ``data: <text>\\r\\n\\r\\n`` with no ``event:``
    prefix. Old clients that only listen to ``EventSource.onmessage`` rely on
    this contract — breaking it breaks the chat."""
    app = make_app_with_events([
        {"type": "response_chunk", "content": "Hola"},
        {"type": "response_chunk", "content": " mundo"},
    ])
    body = await post_chat_body(app)
    # Two unnamed data lines for two tokens
    assert body.count(b"data: Hola\r\n\r\n") == 1
    assert body.count(b"data:  mundo\r\n\r\n") == 1
    # No event prefix on response_chunk path
    assert b"event:" not in body


# ── cache_hit path: byte-perfect reproduction of the user-reported bug ──────


@pytest.mark.asyncio
async def test_cache_hit_path_full_byte_sequence():
    """The exact event sequence chain.py emits when semantic cache hits.
    Produced this scenario in production:

        I am sad
        {"language": "en", ...}        ← agent_start (leaked text in old client)
        {}                              ← cache_hit  (leaked)
        I'm sorry to hear...
        {"stop_reason": "cache", ...}  ← agent_done  (leaked)

    A correctly-named-event format makes the leak impossible at the source.
    """
    app = make_app_with_events([
        {"type": "agent_start", "language": "en", "has_image": False},
        {"type": "cache_hit"},
        {"type": "response_chunk",
         "content": "I'm sorry.\n\nTell me more about what's making you feel this way?"},
        {"type": "agent_done", "stop_reason": "cache", "latency_ms": 86},
    ])
    body = await post_chat_body(app)

    # Named events clearly demarcated
    assert b"event: agent_start\r\n" in body
    assert b"event: cache_hit\r\n" in body
    assert b"event: agent_done\r\n" in body

    # Token content present as unnamed data lines (sse_starlette splits on \n)
    assert b"data: I'm sorry." in body
    assert b"data: Tell me more about what's making you feel this way?" in body

    # Critical: every named-event JSON payload follows an `event:` line, never
    # appears as a bare `data: {"...` (which would be the leaky old format)
    lines = body.split(b"\r\n")
    for i, line in enumerate(lines):
        if line.startswith(b"data: ") and (b'"language"' in line or b'"stop_reason"' in line):
            # If a named-event JSON is in a data line, the previous line MUST be `event:`
            prev = lines[i - 1] if i > 0 else b""
            assert prev.startswith(b"event: "), (
                f"Named-event JSON {line!r} found without preceding `event:` "
                f"line — this is the leaky bug pattern!"
            )


# ── multi-line response_chunk content ────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiline_chunk_content_splits_into_data_lines():
    """sse_starlette splits ``\\n`` inside a value into multiple ``data:``
    lines per spec — test confirms this behavior is preserved so clients
    can reassemble."""
    app = make_app_with_events([
        {"type": "response_chunk", "content": "line1\nline2\nline3"},
    ])
    body = await post_chat_body(app)
    assert b"data: line1\r\n" in body
    assert b"data: line2\r\n" in body
    assert b"data: line3\r\n" in body
    # All three lines belong to the same event (one terminator after them all)
    # Find the position of `data: line1`, then `\r\n\r\n` (terminator) AFTER `line3`
    idx_l1 = body.index(b"data: line1\r\n")
    idx_l3 = body.index(b"data: line3\r\n", idx_l1)
    # Between l1 and l3 there must NOT be a `\r\n\r\n` terminator
    assert b"\r\n\r\n" not in body[idx_l1:idx_l3]


# ── guard_blocked path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_blocked_event_present_with_pattern():
    app = make_app_with_events([
        {"type": "agent_start", "language": "es", "has_image": False},
        {"type": "guard_blocked", "pattern": "ignore previous"},
        {"type": "response_chunk", "content": "Lo siento."},
        {"type": "agent_done", "stop_reason": "guard", "latency_ms": 5},
    ])
    body = await post_chat_body(app)
    assert b"event: guard_blocked\r\n" in body
    assert b'"pattern": "ignore previous"' in body
    assert b'"stop_reason": "guard"' in body


# ── content-type header ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_response_content_type_is_event_stream():
    app = make_app_with_events([
        {"type": "response_chunk", "content": "ok"},
    ])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/chat", json={"user_id": "u1", "message": "hi"})
        assert "text/event-stream" in resp.headers.get("content-type", "")
