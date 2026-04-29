"""User profile endpoint — first-run onboarding after Google sign-in.

The frontend calls this once after the Google ID-token has been verified
(see ``auth.py``). It captures the minimum essential profile that lets
Alma greet by name and tailor copy by age — without re-asking on every
new session.

What this endpoint stores (all in the ``interaction_prefs`` memory layer):

  entry_key=name        → {"preference": "Se llama X", "name": X}
  entry_key=age_range   → {"preference": "Tiene Y", "age_range": Y}
  entry_key=phone       → {"preference": "Celular registrado", "phone": +51..., "phone_last4": "...XXXX"}

The phone field is OPTIONAL. We store the raw value because Cloud SQL is
encrypted at rest and we may need it later for proactive Web Push fallback
(WS-D.3) — NOT for marketing. A ``phone_last4`` companion field is also
stored so frontend can display ``…1234`` without the agent ever returning
the full number outside this account's owner.

WS-D.2 will merge any pre-existing anonymous-UUID memory into the
``google_<sub>`` identity used here, so a user who chats first and signs
in second loses nothing.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["users"])


_ALLOWED_AGE_RANGES: tuple[str, ...] = (
    "under_18",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-plus",
    "prefer_not_to_say",
)


class UserProfileRequest(BaseModel):
    user_id: str = Field(..., description="Canonical google_<sub> after auth/google verify")
    name: str = Field(..., min_length=1, max_length=80)
    age_range: Literal[
        "under_18", "18-24", "25-34", "35-44", "45-54", "55-plus", "prefer_not_to_say"
    ]
    phone: str | None = Field(default=None, max_length=24)


class UserProfileResponse(BaseModel):
    ok: bool


def _age_range_label_es(age_range: str) -> str:
    return {
        "under_18": "menor de 18 años",
        "18-24": "entre 18 y 24 años",
        "25-34": "entre 25 y 34 años",
        "35-44": "entre 35 y 44 años",
        "45-54": "entre 45 y 54 años",
        "55-plus": "55 años o más",
        "prefer_not_to_say": "edad no compartida",
    }.get(age_range, age_range)


def _normalize_phone(raw: str) -> tuple[str, str] | None:
    """Strip spaces/dashes, keep leading +. Return ``(full, last4)`` or None.

    We don't validate by country code — the user's input is the source of
    truth. We just refuse anything with fewer than 7 digits (clearly not a
    phone) so empty submissions don't pollute the layer.
    """
    cleaned = re.sub(r"[^\d+]", "", raw or "")
    digits_only = re.sub(r"\D", "", cleaned)
    if len(digits_only) < 7:
        return None
    last4 = digits_only[-4:]
    return cleaned, last4


@router.post("/users/profile", response_model=UserProfileResponse)
async def upsert_profile(req: UserProfileRequest, request: Request) -> UserProfileResponse:
    """Store the onboarding profile under ``interaction_prefs``.

    Idempotent — calling twice updates the same three entry_keys. The
    user_id MUST already be a verified ``google_<sub>`` (the frontend
    only calls this AFTER auth/google succeeds), so we do not re-verify
    a token here. WS-D.4 may later add cross-channel auth proofs.
    """
    if not req.user_id.startswith("google_") and not req.user_id.startswith("tg_"):
        # Defense-in-depth: anonymous UUIDs should not store a profile yet.
        # If they want to, they sign in first.
        raise HTTPException(status_code=400, detail="Profile requires authenticated identity")

    if req.age_range not in _ALLOWED_AGE_RANGES:
        raise HTTPException(status_code=422, detail="Unsupported age_range")

    mcp = request.app.state.mcp_client

    name_clean = req.name.strip()
    if not name_clean:
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    await mcp.upsert_memory(
        req.user_id,
        "interaction_prefs",
        {
            "entry_key": "name",
            "preference": f"Se llama {name_clean}",
            "name": name_clean,
        },
    )

    await mcp.upsert_memory(
        req.user_id,
        "interaction_prefs",
        {
            "entry_key": "age_range",
            "preference": f"Tiene {_age_range_label_es(req.age_range)}",
            "age_range": req.age_range,
        },
    )

    if req.phone:
        normalized = _normalize_phone(req.phone)
        if normalized is not None:
            full, last4 = normalized
            await mcp.upsert_memory(
                req.user_id,
                "interaction_prefs",
                {
                    "entry_key": "phone",
                    "preference": "Tiene un celular registrado para contacto opcional futuro",
                    "phone": full,
                    "phone_last4": last4,
                },
            )

    logger.info("Profile upsert ok for user_id=%s age=%s phone=%s",
                req.user_id, req.age_range, "yes" if req.phone else "no")

    return UserProfileResponse(ok=True)
