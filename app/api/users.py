"""User preferences endpoint — feature opt-ins after Google sign-in.

The frontend opens a "Mi perfil" modal from the sidebar. Two yes/no questions:

  1. Do you want Alma to remember between sessions? (consent to persistence)
  2. Do you want Alma to write to you when you go silent? Pick a channel.

The reframe — from a 3-field data form ("name + age + phone") to two
feature consents — was driven by user testing of WS-D.5: María flagged
the post-first-message data ask as breaking a moment of vulnerability.
The same fields exist conceptually but now they're surfaced contextually:

  · `name` already comes from Google (no input needed).
  · `age_range` is no longer asked — the agent calibrates tone from the
    first message.
  · `phone` is only asked when the user picked "SMS" as the proactive
    channel — captured in a follow-up flow, not in this endpoint.

Storage stays under the existing ``interaction_prefs`` memory layer,
with two new entry_keys:

  entry_key=remember_consent → {"preference": "Acepta que Alma recuerde…", "remember": true|false}
  entry_key=proactive_channel → {"preference": "Prefiere recibir mensajes por <channel>", "channel": "<...>"}

The legacy ``age_range`` and ``phone`` fields remain accepted (optional)
so the endpoint is non-breaking for any caller that already submitted
WS-D.5-style payloads — but new clients should only send the new shape.
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
    """Profile payload — supports both the new feature-opt-in shape and
    the legacy WS-D.5 shape (age_range + phone). Only ``user_id`` and
    ``name`` are strictly required; everything else is optional.
    """
    user_id: str = Field(..., description="Canonical google_<sub> or tg_<id>")
    name: str = Field(..., min_length=1, max_length=80)
    # New (E.8): two feature opt-ins. Both default to False/none, meaning
    # the user can save just a name without committing to anything else.
    remember_consent: bool | None = Field(
        default=None,
        description="True if user opted into cross-session persistence",
    )
    proactive_channel: Literal["none", "push", "telegram", "sms"] | None = Field(
        default=None,
        description="Channel the user wants Alma to use for proactive check-ins, or 'none'",
    )
    # Legacy (WS-D.5): kept optional so any deployed client that still
    # submits these fields keeps working. New clients should not send them.
    age_range: Literal[
        "under_18", "18-24", "25-34", "35-44", "45-54", "55-plus", "prefer_not_to_say"
    ] | None = Field(default=None)
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


_CHANNEL_LABEL_ES: dict[str, str] = {
    "none": "no quiere mensajes proactivos por ningún canal",
    "push": "acepta recibir mensajes por push del navegador",
    "telegram": "acepta recibir mensajes por Telegram",
    "sms": "acepta recibir mensajes por SMS",
}


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
    """Store the user's profile / feature consents under ``interaction_prefs``.

    Idempotent — calling twice updates the same entry_keys. The user_id
    MUST already be a verified ``google_<sub>`` or ``tg_<id>`` (the
    frontend only calls this AFTER auth/google succeeds, or from the
    Telegram bot), so we do not re-verify a token here.
    """
    if not req.user_id.startswith("google_") and not req.user_id.startswith("tg_"):
        # Defense-in-depth: anonymous UUIDs should not store a profile yet.
        raise HTTPException(status_code=400, detail="Profile requires authenticated identity")

    mcp = request.app.state.mcp_client

    name_clean = req.name.strip()
    if not name_clean:
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    # 1) Always store the canonical name from Google (or whatever the user typed).
    await mcp.upsert_memory(
        req.user_id,
        "interaction_prefs",
        {
            "entry_key": "name",
            "preference": f"Se llama {name_clean}",
            "name": name_clean,
        },
    )

    # 2) Remember-consent — only stored when the field is explicitly set.
    if req.remember_consent is not None:
        await mcp.upsert_memory(
            req.user_id,
            "interaction_prefs",
            {
                "entry_key": "remember_consent",
                "preference": (
                    "Acepta que Alma recuerde lo conversado entre sesiones"
                    if req.remember_consent
                    else "Prefiere que cada sesión sea nueva sin memoria entre llamadas"
                ),
                "remember": bool(req.remember_consent),
            },
        )

    # 3) Proactive-channel — same pattern. "none" is a valid explicit
    # choice (user opted OUT of being contacted) and we record that too,
    # so the scheduler knows not to bother them.
    if req.proactive_channel is not None:
        await mcp.upsert_memory(
            req.user_id,
            "interaction_prefs",
            {
                "entry_key": "proactive_channel",
                "preference": _CHANNEL_LABEL_ES.get(
                    req.proactive_channel, f"canal {req.proactive_channel}"
                ),
                "channel": req.proactive_channel,
            },
        )

    # 4) Legacy fields — kept for backward-compat with WS-D.5 clients.
    if req.age_range is not None and req.age_range in _ALLOWED_AGE_RANGES:
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

    logger.info(
        "Profile upsert ok for user_id=%s remember=%s channel=%s legacy_age=%s legacy_phone=%s",
        req.user_id,
        req.remember_consent,
        req.proactive_channel,
        req.age_range,
        "yes" if req.phone else "no",
    )

    return UserProfileResponse(ok=True)
