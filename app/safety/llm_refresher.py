"""WS-H.4 — Background LLM probe refresher.

Runs every interval_s seconds and writes a snapshot dict to
app.state.llm_health. The /ready endpoint reads this dict only — it never
awaits the LLM directly. Failures are swallowed: a failed probe writes
{ok: False, error: ...} and the loop continues on the next tick.
"""
import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI

from app.agent.llm import _ping_anthropic, _ping_gemini, PROVIDER_CHAIN

logger = logging.getLogger(__name__)


async def probe_once() -> dict[str, Any]:
    """Probe the provider chain once. Returns a snapshot dict."""
    first_failure: str | None = None
    for provider, model, _max_tokens in PROVIDER_CHAIN:
        try:
            if provider == "anthropic":
                ok, err = await _ping_anthropic(model)
            else:
                ok, err = await _ping_gemini(model)
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {str(exc)[:140]}"

        if ok:
            return {
                "ok": True,
                "provider": provider,
                "model": model,
                "error": None,
                "last_probed_at": time.time(),
            }
        if first_failure is None:
            first_failure = f"{provider}/{model}: {err}"

    return {
        "ok": False,
        "provider": None,
        "model": None,
        "error": first_failure or "no provider configured",
        "last_probed_at": time.time(),
    }


async def refresher_loop(app: FastAPI, interval_s: float = 60.0) -> None:
    """Long-lived task; updates app.state.llm_health every interval_s.

    Survives probe failures — exceptions are logged and the loop continues."""
    while True:
        try:
            snapshot = await probe_once()
            app.state.llm_health = snapshot
            if not snapshot["ok"]:
                logger.warning("LLM probe degraded: %s", snapshot["error"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("LLM refresher tick failed: %s", exc)
        await asyncio.sleep(interval_s)
