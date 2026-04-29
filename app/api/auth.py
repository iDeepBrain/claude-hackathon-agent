"""Google OAuth ID-token verification + anonymous-to-Google identity link.

The frontend obtains an ID token via Google Identity Services (GIS),
sends it here. We verify the token signature against Google's certs
and extract the user's ``google_sub``, ``email``, and ``name``. The
frontend then swaps ``localStorage.alma_user_id`` from the anonymous
UUID to ``"google_<sub>"``, which becomes the persistent identity
for memory storage going forward.

WS-D.2 (revised) added two endpoints/extensions:

- ``POST /auth/google`` accepts an optional ``anon_uuid`` and returns
  ``prior_anon_session`` describing whether the UI should prompt the
  user to link the anonymous conversation.

- ``POST /auth/link-anonymous`` performs the linking decision (keep
  or reset) by re-verifying the id_token and calling the MCP tool
  ``link_anonymous_to_account_tool``. Single transaction in the MCP
  side; idempotent on retry.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])

# WS-D.2 — gates for the rejoin prompt
_PROMPT_MIN_MESSAGES = 3
_CRISIS_AUTOKEEP_THRESHOLD = 0.4
_ANON_UUID_RE = re.compile(r"^uuid_[0-9a-fA-F-]{8,}$")
_RESERVED_ANON_UUIDS = {"demo_mateo"}


class GoogleAuthRequest(BaseModel):
    id_token: str
    anon_uuid: str | None = None  # WS-D.2 — optional


class PriorAnonSession(BaseModel):
    uuid: str
    message_count: int
    should_prompt: bool
    auto_kept: bool


class GoogleAuthResponse(BaseModel):
    user_id: str
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None
    prior_anon_session: PriorAnonSession | None = None  # WS-D.2


class LinkAnonymousRequest(BaseModel):
    id_token: str
    anon_uuid: str
    action: Literal["keep", "reset"]


class LinkAnonymousResponse(BaseModel):
    ok: bool
    action: Literal["keep", "reset"]
    merged_layers: int = Field(0, ge=0)


def _expected_audience() -> str:
    return os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")


def _validate_anon_uuid(anon_uuid: str) -> None:
    """Raise HTTPException if the uuid is malformed or reserved."""
    if not anon_uuid or not _ANON_UUID_RE.match(anon_uuid):
        raise HTTPException(status_code=400, detail="Invalid anon_uuid format")
    if anon_uuid in _RESERVED_ANON_UUIDS:
        raise HTTPException(status_code=400, detail="anon_uuid is reserved")
    if anon_uuid.startswith(("google_", "tg_")):
        raise HTTPException(status_code=400, detail="anon_uuid must be anonymous")


def _verify_google_token(token: str) -> dict:
    """Verify the id_token against Google's certs. Raises 401 on failure."""
    expected_aud = _expected_audience()
    if not expected_aud:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth disabled (GOOGLE_OAUTH_CLIENT_ID env unset)",
        )
    try:
        info = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            expected_aud,
        )
    except ValueError as exc:
        logger.info("Rejected Google id_token: %s", exc)
        raise HTTPException(status_code=401, detail=f"Invalid id_token: {exc}")
    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
    return info


async def _evaluate_prior_anon_session(
    request: Request, anon_uuid: str, google_sub: str
) -> PriorAnonSession | None:
    """Decide what to do with the anonymous session.

    Returns None if the uuid has no data.
    Returns PriorAnonSession with should_prompt=True for normal cases.
    Returns PriorAnonSession with auto_kept=True after silent server-side
    link when crisis_score > threshold.
    """
    mcp = getattr(request.app.state, "mcp_client", None)
    if mcp is None:
        return None

    memory = await mcp.get_memory(anon_uuid)
    # message_count: rough proxy via mood_history + mentioned_events length
    message_count = sum(
        len(memory.get(k, [])) for k in
        ("mood_history", "mentioned_events", "habits", "interaction_prefs")
    )
    if message_count == 0:
        return None

    # crisis_score: peek the most recent mood_history entry's score, fallback 0
    mood_entries = memory.get("mood_history", [])
    crisis_score = 0.0
    for entry in mood_entries:
        content = entry.get("content", {}) if isinstance(entry, dict) else {}
        score = content.get("crisis_score", 0)
        if isinstance(score, (int, float)) and score > crisis_score:
            crisis_score = float(score)

    if crisis_score > _CRISIS_AUTOKEEP_THRESHOLD:
        # Silent preserve. Don't ask the user during emotional distress.
        try:
            await mcp.link_anonymous_to_account(anon_uuid, google_sub, "keep")
            logger.info(
                "Auto-kept anon=%s under google=%s (crisis_score=%.2f)",
                anon_uuid, google_sub, crisis_score,
            )
            return PriorAnonSession(
                uuid=anon_uuid,
                message_count=message_count,
                should_prompt=False,
                auto_kept=True,
            )
        except Exception as exc:
            logger.exception("Auto-keep failed: %s", exc)
            return None

    return PriorAnonSession(
        uuid=anon_uuid,
        message_count=message_count,
        should_prompt=message_count > _PROMPT_MIN_MESSAGES,
        auto_kept=False,
    )


@router.post("/auth/google", response_model=GoogleAuthResponse)
async def auth_google(req: GoogleAuthRequest, request: Request) -> GoogleAuthResponse:
    info = _verify_google_token(req.id_token)
    sub = info["sub"]
    google_user_id = f"google_{sub}"

    prior = None
    if req.anon_uuid:
        try:
            _validate_anon_uuid(req.anon_uuid)
        except HTTPException as exc:
            # Reject only the link signal, not the login itself.
            logger.info("Login proceeds; anon_uuid invalid: %s", exc.detail)
            prior = None
        else:
            prior = await _evaluate_prior_anon_session(
                request, req.anon_uuid, google_user_id
            )

    logger.info("Google auth success for sub=%s anon=%s prompt=%s",
                sub, req.anon_uuid, prior.should_prompt if prior else None)
    return GoogleAuthResponse(
        user_id=google_user_id,
        email=info.get("email", ""),
        email_verified=bool(info.get("email_verified", False)),
        name=info.get("name"),
        picture=info.get("picture"),
        prior_anon_session=prior,
    )


@router.post("/auth/link-anonymous", response_model=LinkAnonymousResponse)
async def auth_link_anonymous(
    req: LinkAnonymousRequest, request: Request
) -> LinkAnonymousResponse:
    info = _verify_google_token(req.id_token)
    google_sub = f"google_{info['sub']}"

    _validate_anon_uuid(req.anon_uuid)

    mcp = getattr(request.app.state, "mcp_client", None)
    if mcp is None:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    result = await mcp.link_anonymous_to_account(req.anon_uuid, google_sub, req.action)
    if not result.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"link failed: {result.get('error', 'unknown')}",
        )
    return LinkAnonymousResponse(
        ok=True,
        action=req.action,
        merged_layers=int(result.get("merged_layers", 0)),
    )
