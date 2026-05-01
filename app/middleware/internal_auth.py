"""Internal bearer-token guard for the public Cloud Run agent URL.

The agent runs with ``--allow-unauthenticated`` so the web nginx (and the
Telegram bot) can reach it without GCP IAM. This middleware closes the
direct-curl bypass: every protected endpoint requires
``Authorization: Bearer <ALMA_INTERNAL_TOKEN>`` where the token lives
in Secret Manager and is mounted into nginx, the agent, and the bot.

**Fail-open by design.** If the token env var is empty/unset, the
middleware lets every request through. This lets us roll out the
secret across services without a synchronized cut-over — once the
secret is bound on the agent, the check turns on automatically.

Endpoints that MUST stay open (no token check):

* ``/health`` — Cloud Run TCP/HTTP probe target.
* ``/api/v1/ready`` — deep health probe.
* ``/api/v1/config`` — public client config (the unauthenticated web
  page calls this on first paint).
* ``/cron/*`` — Cloud Scheduler endpoints already use their own
  ``X-Cloud-Scheduler-Token`` check.
* ``/docs``, ``/openapi.json``, ``/redoc`` — FastAPI docs.

CORS preflights (``OPTIONS``) are also exempt — preflights run before
the browser attaches Authorization headers.
"""
from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

_SKIP_PATHS = {
    "/health",
    "/api/v1/ready",
    "/api/v1/config",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
}
_SKIP_PREFIXES = ("/cron/",)


class InternalAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str | None, environment: str | None = None):
        super().__init__(app)
        self._token = (token or "").strip()
        self._environment = (environment or "").strip().lower()
        # Enforcement is only on when BOTH conditions hold:
        #   1. A token is bound (ALMA_INTERNAL_TOKEN env var set).
        #   2. We are in a non-local environment (ALMA_ENV != "local").
        # Local docker compose binds the token for parity with prod so the
        # value is auditable in .env.local, but does not need bearer
        # enforcement — the local stack is single-tenant.
        self._enforce = bool(self._token) and self._environment != "local"
        if self._enforce:
            logger.info("InternalAuthMiddleware: bearer enforcement ENABLED (env=%s)", self._environment or "<unset>")
        else:
            reason = "no token" if not self._token else f"env={self._environment} (local-skip)"
            logger.info("InternalAuthMiddleware: fail-open (%s)", reason)

    def _is_exempt(self, request: Request) -> bool:
        if request.method == "OPTIONS":
            return True
        path = request.url.path
        if path in _SKIP_PATHS:
            return True
        for prefix in _SKIP_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        if not self._enforce or self._is_exempt(request):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "missing_bearer", "detail": "Authorization required"},
            )
        if auth[7:].strip() != self._token:
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_bearer", "detail": "Authorization invalid"},
            )
        return await call_next(request)
