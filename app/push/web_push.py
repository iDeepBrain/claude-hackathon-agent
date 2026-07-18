"""WS-D.3 — Web Push (VAPID standard).

Sends notifications directly to a browser's push service using the standard
Web Push Protocol via the pywebpush library. No Firebase, no third-party
service-broker between Alma and the user's browser.

A push call returns one of three outcomes:
  - "sent":     201 Created from the push service. Notification on the way.
  - "expired":  410 Gone. Subscription is no longer valid; caller should
                clear push_subscription on the user row to stop retries.
  - "failed":   any other error. Caller should NOT clear the subscription;
                a transient error today does not mean the endpoint is dead.

The module gates on three env vars: VAPID_PRIVATE_KEY, VAPID_PUBLIC_KEY,
VAPID_SUBJECT. If any is unset, ``send_push`` raises ``WebPushDisabled`` —
the scheduler must check ``is_configured()`` before calling.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

PushOutcome = Literal["sent", "expired", "failed"]


class WebPushDisabled(RuntimeError):
    """Raised when send_push() is called but VAPID env vars are not configured."""


def is_configured() -> bool:
    return all(
        os.environ.get(k)
        for k in ("VAPID_PRIVATE_KEY", "VAPID_PUBLIC_KEY", "VAPID_SUBJECT")
    )


def get_public_key() -> str | None:
    """Returns the URL-safe base64 VAPID public key, or None if disabled."""
    return os.environ.get("VAPID_PUBLIC_KEY") or None


def _claims() -> dict[str, str]:
    return {"sub": os.environ["VAPID_SUBJECT"]}


_CACHED_PRIVATE_KEY: str | None = None


def _vapid_private_key_for_pywebpush() -> str:
    """Return the VAPID private key in the exact form pywebpush expects:
    URL-safe base64 of the DER-encoded PKCS8 key.

    Why this exists: pywebpush calls Vapid.from_string(...) which auto-detects
    the format via heuristics on the string. PEM keys are accepted in theory,
    but Cloud Run's env-var injection can mangle multi-line PEM (escaping
    newlines as \\n, stripping BEGIN/END headers, etc.) — when that happens,
    py_vapid silently falls through to from_der() and crashes with
    'invalid length / could not deserialize'.

    Bullet-proof fix: detect PEM ourselves, decode via the cryptography lib
    (which is already a transitive dep), and re-encode as DER-b64 single line.
    Cached so we only do the expensive parse once per process.
    """
    global _CACHED_PRIVATE_KEY
    if _CACHED_PRIVATE_KEY is not None:
        return _CACHED_PRIVATE_KEY

    raw = os.environ["VAPID_PRIVATE_KEY"].strip()

    # Some env injectors deliver multi-line values as a single line with
    # literal "\n" sequences. Normalize before format detection.
    if "\\n" in raw and "\n" not in raw:
        raw = raw.replace("\\n", "\n")

    if "BEGIN" in raw:
        # PEM → load via cryptography → re-encode as DER → URL-safe b64.
        import base64

        from cryptography.hazmat.primitives import serialization

        key = serialization.load_pem_private_key(raw.encode(), password=None)
        der = key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        _CACHED_PRIVATE_KEY = base64.urlsafe_b64encode(der).rstrip(b"=").decode("ascii")
    else:
        # Already in DER-b64 (single line) form.
        _CACHED_PRIVATE_KEY = raw

    logger.info("VAPID private key normalized for pywebpush (len=%d)", len(_CACHED_PRIVATE_KEY))
    return _CACHED_PRIVATE_KEY


async def send_push(subscription: dict, payload: dict) -> PushOutcome:
    """Send one notification. Returns 'sent' / 'expired' / 'failed'.

    pywebpush is sync — we run it in a thread to avoid blocking the loop.
    """
    if not is_configured():
        raise WebPushDisabled("VAPID env vars not set")

    return await asyncio.to_thread(_send_blocking, subscription, payload)


def _send_blocking(subscription: dict, payload: dict) -> PushOutcome:
    from pywebpush import WebPushException, webpush  # imported lazy

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=_vapid_private_key_for_pywebpush(),
            vapid_claims=_claims(),
            timeout=10,
        )
        return "sent"
    except WebPushException as exc:
        # Some pywebpush versions expose .response, others .reason
        status = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
        if status == 410 or status == 404:
            logger.info("Push subscription expired (HTTP %s)", status)
            return "expired"
        logger.warning("WebPushException: status=%s message=%s", status, str(exc)[:140])
        return "failed"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Push send failed: %s", exc)
        return "failed"


def build_payload(slot: str, title: str, body: str, url: str = "/demo.html") -> dict[str, Any]:
    """Standard payload shape consumed by sw.js."""
    return {"title": title, "body": body, "slot": slot, "url": url}
