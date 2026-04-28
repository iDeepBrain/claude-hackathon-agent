"""Google OAuth ID-token verification.

The frontend obtains an ID token via Google Identity Services (GIS),
sends it here. We verify the token signature against Google's certs
and extract the user's ``google_sub``, ``email``, and ``name``. The
frontend then swaps ``localStorage.alma_user_id`` from the anonymous
UUID to ``"google_<sub>"``, which becomes the persistent identity
for memory storage going forward.

Migration of any pre-existing anonymous memory to the new
``google_<sub>`` identity is WS-D.2's job — this endpoint only
verifies and returns the canonical identity. By WS-D.3 the same
identity gets a Web Push subscription so Alma can reach out.

Pre-deploy:
- Create an OAuth Client ID in Google Cloud Console (type: "Web").
- Set ``GOOGLE_OAUTH_CLIENT_ID`` env var on the agent service.
- Add the deployed origin (https://alma-bot.com) to the client's
  authorized JavaScript origins.

Without ``GOOGLE_OAUTH_CLIENT_ID`` set, the endpoint returns 503 —
matches the demo seed endpoint pattern, fails closed by design.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


class GoogleAuthRequest(BaseModel):
    id_token: str


class GoogleAuthResponse(BaseModel):
    user_id: str  # canonical "google_<sub>"
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None


def _expected_audience() -> str:
    return os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")


@router.post("/auth/google", response_model=GoogleAuthResponse)
async def auth_google(req: GoogleAuthRequest) -> GoogleAuthResponse:
    """Verify a Google ID token and return the canonical user_id.

    The endpoint NEVER returns a user_id without first verifying the
    token signature against Google's certs. The verify step also
    checks the audience matches our client_id — preventing token
    reuse from other apps with the same provider.
    """
    expected_aud = _expected_audience()
    if not expected_aud:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth disabled (GOOGLE_OAUTH_CLIENT_ID env unset)",
        )

    try:
        info = google_id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            expected_aud,
        )
    except ValueError as exc:
        logger.info("Rejected Google id_token: %s", exc)
        raise HTTPException(status_code=401, detail=f"Invalid id_token: {exc}")

    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")

    logger.info("Google auth success for sub=%s", sub)
    return GoogleAuthResponse(
        user_id=f"google_{sub}",
        email=info.get("email", ""),
        email_verified=bool(info.get("email_verified", False)),
        name=info.get("name"),
        picture=info.get("picture"),
    )
