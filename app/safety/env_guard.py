"""Startup-time environment sanity check.

Cross-validates ALMA_ENV against the host parts of DATABASE_URL /
REDIS_URL / etc. so a misconfigured deploy fails LOUD at boot rather
than silently serving production traffic against a local DB or vice-
versa. WS-H.2 — defense-in-depth on top of the URL whitelist already
enforced inside ``scripts/reset_local.py``.

The validation is intentionally strict:

  · ``ALMA_ENV=local``  → URL host MUST be in the local whitelist
                          (``localhost``, ``127.0.0.1``, ``postgres``,
                          ``redis``, ``::1``).
  · ``ALMA_ENV=prod``   → URL host MUST NOT be in the local whitelist.
                          Empty host is allowed (Cloud SQL unix socket
                          format ``?host=/cloudsql/...``).
  · unset / blank       → refuse — production should set the var
                          explicitly via cloudbuild.yaml.

Mismatch raises ``EnvMismatchError`` which the caller (lifespan or
top-level startup code) propagates up. With an unhandled exception in
the FastAPI lifespan, the app refuses to serve traffic — Cloud Run
marks the revision as failed and previous revision keeps serving.
"""
from __future__ import annotations

import logging
import os
import sys
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _audit(msg: str) -> None:
    """WS-H.5 — emit audit lines to stderr in addition to logger.info().

    Cloud Run captures stderr unconditionally. The agent runs the guard
    inside lifespan (after uvicorn configures logging), so logger.info()
    works here today — but we mirror MCP's behavior to keep the two
    parallel modules identical and to harden against future call sites
    that might run before logging is set up."""
    print(msg, file=sys.stderr, flush=True)
    logger.info(msg)


# Hosts considered safe ONLY for local development. Identical to the
# whitelist used by reset_local.py — kept in sync intentionally.
LOCAL_HOSTS: frozenset[str] = frozenset({
    "localhost",
    "127.0.0.1",
    "postgres",  # docker-compose service name
    "redis",     # docker-compose service name
    "::1",
})


class EnvMismatchError(RuntimeError):
    """Raised when ALMA_ENV does not match a connection URL's host."""


def _read_env(env_var: str = "ALMA_ENV") -> str:
    return os.getenv(env_var, "").strip().lower()


def _read_host(url: str) -> str:
    """Extract the hostname from a SQLAlchemy / Redis / generic URL.

    Returns an empty string for URLs without a hostname (e.g. Cloud SQL
    unix socket: ``postgresql+asyncpg://user:pass@/dbname?host=/cloudsql/x``).
    """
    if not url:
        return ""
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        # Some non-standard URL forms can throw; treat as empty (will
        # then trigger the prod-only "host required" check downstream).
        return ""


def validate_url_against_env(
    url_var: str,
    label: str = "",
    *,
    env_var: str = "ALMA_ENV",
    allow_empty_host_in_prod: bool = True,
) -> None:
    """Validate one URL env var against ALMA_ENV. Raise on mismatch.

    Parameters
    ----------
    url_var
        Env var name to read (e.g. ``"DATABASE_URL"``, ``"REDIS_URL"``).
    label
        Human-readable label for log lines (e.g. ``"postgres"``).
        Defaults to ``url_var``.
    env_var
        Env var name for the env mode. Defaults to ``ALMA_ENV``.
    allow_empty_host_in_prod
        When True (default), an empty hostname is treated as valid in
        prod (Cloud SQL unix socket case). Set False to require an
        explicit host even in prod.
    """
    env = _read_env(env_var)
    label = label or url_var
    url = os.getenv(url_var, "").strip()

    if not env:
        raise EnvMismatchError(
            f"{env_var} is unset. Set it explicitly: 'local' for the "
            f"docker-compose stack, 'prod' for Cloud Run (cloudbuild.yaml)."
        )

    if env not in ("local", "prod"):
        raise EnvMismatchError(
            f"{env_var} must be 'local' or 'prod', got {env!r}."
        )

    if not url:
        raise EnvMismatchError(
            f"{url_var} is unset — required to validate the {label} connection."
        )

    host = _read_host(url)

    if env == "local":
        if host not in LOCAL_HOSTS:
            raise EnvMismatchError(
                f"ALMA_ENV=local but {url_var} host {host!r} is NOT a "
                f"local host. Allowed: {sorted(LOCAL_HOSTS)}. Refusing to "
                f"connect — a local stack should never reach a remote {label}."
            )
        return  # OK

    # env == "prod"
    if host in LOCAL_HOSTS:
        raise EnvMismatchError(
            f"ALMA_ENV=prod but {url_var} host {host!r} is the LOCAL "
            f"whitelist. This deploy would connect production traffic to "
            f"a local-style {label} — refusing to start."
        )
    if not host and not allow_empty_host_in_prod:
        raise EnvMismatchError(
            f"ALMA_ENV=prod and {url_var} has no host. If using a Cloud "
            f"SQL unix socket, set allow_empty_host_in_prod=True."
        )
    return  # OK


def validate_all_or_raise(*pairs: tuple[str, str]) -> None:
    """Validate multiple URL env vars in one call. Raises on first mismatch.

    Each ``pair`` is ``(url_env_var, label)``. Logs a single OK line per
    URL on success so deploys leave a clear audit trail in Cloud Run logs.
    """
    env = _read_env()
    if not env:
        # Fall through to per-URL validator which produces the canonical error
        if pairs:
            validate_url_against_env(*pairs[0])
        return
    _audit(f"Env-guard: ALMA_ENV={env}")
    for url_var, label in pairs:
        validate_url_against_env(url_var, label)
        _audit(f"Env-guard OK: {label} host validated for env={env}")
