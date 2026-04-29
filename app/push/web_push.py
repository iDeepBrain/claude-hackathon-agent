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
            vapid_private_key=os.environ["VAPID_PRIVATE_KEY"],
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
