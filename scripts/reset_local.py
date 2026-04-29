"""Single-source-of-truth local reset.

Wipes ALL data in the local Postgres + Redis stack and (optionally)
re-seeds the canonical demo persona "Mateo" so the demo flow has
populated state on first open. Designed to be the ONE place to add
table-cleanup logic — if a future migration adds a table, the cleanup
goes here, not scattered across ad-hoc psql DELETEs.

## Safety guards (refusing to run when ANY fails)

  · ``ALMA_ENV`` env var must be exactly ``local``. Production deploys
    set ``ALMA_ENV=prod`` (cloudbuild.yaml), so this script aborts.
  · ``DATABASE_URL`` host must match a known-local pattern: ``localhost``,
    ``127.0.0.1``, or the docker-compose service name ``postgres``.
    Refuses against Cloud SQL unix sockets / public IPs.
  · ``REDIS_URL`` host must match the same pattern.

Either guard refusing aborts the script with a non-zero exit code and a
descriptive error — no partial wipe, no surprises.

## Usage

```bash
# From the agent container (recommended — env vars + network ready):
docker compose exec agent python scripts/reset_local.py
docker compose exec agent python scripts/reset_local.py --no-seed
docker compose exec agent python scripts/reset_local.py --user-id google_smoke_42

# From the repo root via Makefile:
make reset
```

## What it does

1. Postgres: ``DELETE FROM`` every ``alma_*`` table in dependency order
2. Redis: ``FLUSHDB`` (semantic cache + sessions + proactive gates)
3. (Optional, default on) Re-seed the canonical Mateo persona via the
   existing ``app.seed_demo`` helper

If ``--user-id`` is provided, only that user's data is wiped — useful
for clearing a smoke test user without touching anyone else's local
state.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse

# When invoked as `python scripts/reset_local.py` from /app, Python only
# puts /app/scripts on sys.path, which makes `import app.seed_demo` fail.
# Adding the parent (the project root) lets the reseed step find the
# package without forcing callers to use `python -m scripts.reset_local`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Hosts considered safe for destructive operations. If DATABASE_URL or
# REDIS_URL doesn't resolve to one of these, the script refuses.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "postgres", "redis", "::1"}

# Tables wiped in order. Foreign-key-free in our schema, but we delete
# memory_layers BEFORE alma_users to keep the order intuitive even if
# constraints get added later.
_TABLES_IN_DELETE_ORDER = [
    "alma_messages",
    "alma_conversations",
    "alma_proactive_log",
    "alma_memory_layers",
    "alma_users",
]


def _bail(reason: str) -> None:
    print(f"ERROR: refusing to reset — {reason}", file=sys.stderr)
    sys.exit(2)


def _check_safety_guards() -> tuple[str, str]:
    """Verify env + host whitelist. Returns (database_url, redis_url) or exits."""
    env = os.getenv("ALMA_ENV", "").strip().lower()
    if env != "local":
        _bail(f"ALMA_ENV must be 'local', got {env!r}. Set ALMA_ENV=local in your .env.")

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        _bail("DATABASE_URL is unset.")
    db_host = urlparse(db_url).hostname or ""
    if db_host not in _LOCAL_HOSTS:
        _bail(f"DATABASE_URL host {db_host!r} is not whitelisted as local. Allowed: {sorted(_LOCAL_HOSTS)}")

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        _bail("REDIS_URL is unset.")
    redis_host = urlparse(redis_url).hostname or ""
    if redis_host not in _LOCAL_HOSTS:
        _bail(f"REDIS_URL host {redis_host!r} is not whitelisted as local. Allowed: {sorted(_LOCAL_HOSTS)}")

    return db_url, redis_url


async def _wipe_postgres(db_url: str, user_id: str | None) -> dict[str, int]:
    """Delete rows. Per-user when ``user_id`` is given, full wipe otherwise.

    Returns a {table: rows_deleted} count for the report.
    """
    # Late imports so safety guards run BEFORE we touch the network.
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    # Normalize asyncpg driver if the URL is plain postgres://
    normalized = db_url
    if normalized.startswith("postgresql://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgresql://"):]
    elif normalized.startswith("postgres://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgres://"):]

    engine = create_async_engine(normalized, echo=False)
    counts: dict[str, int] = {}
    async with engine.begin() as conn:
        for table in _TABLES_IN_DELETE_ORDER:
            if user_id:
                # Only alma_users.user_id is the canonical identifier; the other
                # tables reference it by user_id column too.
                stmt = text(f"DELETE FROM {table} WHERE user_id = :uid")
                params = {"uid": user_id}
            else:
                stmt = text(f"DELETE FROM {table}")
                params = {}
            res = await conn.execute(stmt, params)
            counts[table] = res.rowcount or 0
    await engine.dispose()
    return counts


async def _flush_redis(redis_url: str, user_id: str | None) -> int:
    """FLUSHDB or DEL keys for one user. Returns affected key count."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url)
    if user_id:
        # Targeted: keys keyed by user_id. We scan with the prefixes we know.
        prefixes = [
            f"alma:session:{user_id}",
            f"alma:session:last_activity:{user_id}",
            f"alma:crisis:last:{user_id}",
            f"alma:proactive:last:{user_id}",
            f"alma:tg-chat-for:{user_id}",
            f"alma:cache:user:{user_id}:*",  # scan-pattern
        ]
        deleted = 0
        for p in prefixes:
            if "*" in p:
                async for key in client.scan_iter(match=p, count=100):
                    deleted += await client.delete(key)
            else:
                deleted += await client.delete(p)
        await client.aclose()
        return deleted
    # Full wipe — every key in the current DB
    await client.flushdb()
    await client.aclose()
    return -1  # sentinel for "FLUSHDB ran, no count"


async def _reseed_mateo() -> dict[str, int] | None:
    """Re-populate the canonical demo persona. Returns layer counts or None on failure."""
    try:
        from app.mcp_client.client import MCPClient
        from app.seed_demo import seed_mateo, DEMO_USER_ID_DEFAULT
    except Exception as exc:
        print(f"  (skipped reseed — {exc})")
        return None
    mcp_url = os.environ["MCP_URL"]
    client = MCPClient(mcp_url)
    await client.get_tools()
    counts = await seed_mateo(client, DEMO_USER_ID_DEFAULT)
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe local Alma DB + Redis (singleton-safe).")
    parser.add_argument("--user-id", help="Wipe only this user_id; otherwise full reset.")
    parser.add_argument("--no-seed", action="store_true", help="Skip re-seeding the Mateo persona.")
    parser.add_argument("--no-redis", action="store_true", help="Skip Redis FLUSHDB.")
    args = parser.parse_args()

    db_url, redis_url = _check_safety_guards()
    scope = f"user_id={args.user_id!r}" if args.user_id else "FULL WIPE"
    print(f"reset_local.py — {scope}")
    print(f"  DATABASE_URL host: {urlparse(db_url).hostname}")
    print(f"  REDIS_URL host:    {urlparse(redis_url).hostname}")

    pg_counts = await _wipe_postgres(db_url, args.user_id)
    print("  postgres:")
    for table, n in pg_counts.items():
        print(f"    {table}: {n} rows")

    if not args.no_redis:
        n = await _flush_redis(redis_url, args.user_id)
        if n == -1:
            print("  redis: FLUSHDB ok")
        else:
            print(f"  redis: {n} keys deleted")

    if args.user_id is None and not args.no_seed:
        seed = await _reseed_mateo()
        if seed:
            print(f"  reseed mateo: {seed}")

    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
