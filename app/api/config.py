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
    # Telegram bot username (without the @) — used by the web frontend to
    # build deep-links like https://t.me/<username>?start=<token> when
    # the user picks "Telegram" as proactive channel. Public info, fine
    # to expose. Fallback to env var so deployments can override; if
    # neither is set the frontend hides the channel option.
    tg_username = os.getenv("TELEGRAM_BOT_USERNAME", "").strip()
    if tg_username:
        cfg["telegram_bot_username"] = tg_username.lstrip("@")
    # Environment mode. The frontend uses this to render a "LOCAL DEV"
    # pill so a developer never confuses the local stack with production
    # mid-test. Production sets ALMA_ENV=prod via the Cloud Run env vars
    # in cloudbuild.yaml. We deliberately do NOT default to 'local' —
    # an unset variable surfaces as 'local' (the safer assumption for a
    # reviewer hitting a misconfigured deploy: see the pill, ask). Prod
    # MUST set ALMA_ENV=prod explicitly to suppress the pill.
    env = os.getenv("ALMA_ENV", "").strip().lower()
    if env and env != "prod":
        cfg["env"] = env
    elif not env:
        # Unset — surface a hint so reviewers spot misconfig quickly.
        cfg["env"] = "unset"
    return cfg
