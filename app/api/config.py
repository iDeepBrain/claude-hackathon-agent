"""Public client config — surfaces non-secret env values that the
browser needs to initialize SDK clients (Google Identity Services,
later FCM Web Push).

Intentionally simple GET endpoint. The Client ID is PUBLIC — it goes
into JS bundles anyway. The point of fetching it dynamically (vs
hardcoding in HTML) is so the same demo.html ships across local /
preview / production environments without per-env source forks.

If a value isn't configured server-side, it's omitted from the
response — frontend uses ``if (config.google_oauth_client_id)`` to
gate UI affordances rather than always rendering broken buttons.
"""
from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter(tags=["config"])


@router.get("/config")
async def public_config() -> dict:
    cfg: dict = {}
    google_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    if google_id:
        cfg["google_oauth_client_id"] = google_id
    return cfg
