"""Tests for app.safety.env_guard — WS-H.2 startup sanity check.

Each test sets the env vars explicitly via monkeypatch so the suite
remains hermetic regardless of what the local shell exports.
"""
from __future__ import annotations

import pytest

from app.safety.env_guard import (
    EnvMismatchError,
    LOCAL_HOSTS,
    validate_all_or_raise,
    validate_url_against_env,
)


# ---------- happy paths ----------


def test_local_env_with_local_host_passes(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "local")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    validate_url_against_env("REDIS_URL", "redis")


@pytest.mark.parametrize("host", sorted(LOCAL_HOSTS))
def test_each_whitelisted_host_passes_in_local(monkeypatch, host):
    monkeypatch.setenv("ALMA_ENV", "local")
    # IPv6 addresses must be bracketed in URLs per RFC 3986.
    host_in_url = f"[{host}]" if ":" in host else host
    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:p@{host_in_url}:5432/db")
    validate_url_against_env("DATABASE_URL", "postgres")


def test_prod_env_with_remote_host_passes(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "prod")
    monkeypatch.setenv("REDIS_URL", "redis://10.0.0.1:6379")
    validate_url_against_env("REDIS_URL", "redis")


def test_prod_env_with_cloud_sql_unix_socket_passes(monkeypatch):
    """Cloud SQL unix-socket URLs have no host — allowed by default in prod."""
    monkeypatch.setenv("ALMA_ENV", "prod")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@/db?host=/cloudsql/x")
    validate_url_against_env("DATABASE_URL", "postgres")


# ---------- mismatch refusals ----------


def test_local_env_with_remote_host_refuses(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "local")
    monkeypatch.setenv("REDIS_URL", "redis://prod-redis.example.com:6379")
    with pytest.raises(EnvMismatchError, match="ALMA_ENV=local"):
        validate_url_against_env("REDIS_URL", "redis")


def test_prod_env_with_localhost_refuses(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "prod")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    with pytest.raises(EnvMismatchError, match="ALMA_ENV=prod"):
        validate_url_against_env("REDIS_URL", "redis")


def test_prod_env_with_docker_service_name_refuses(monkeypatch):
    """A prod deploy that accidentally kept docker-compose service names."""
    monkeypatch.setenv("ALMA_ENV", "prod")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@postgres:5432/db")
    with pytest.raises(EnvMismatchError, match="ALMA_ENV=prod"):
        validate_url_against_env("DATABASE_URL", "postgres")


# ---------- env var hygiene ----------


def test_unset_env_refuses(monkeypatch):
    monkeypatch.delenv("ALMA_ENV", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    with pytest.raises(EnvMismatchError, match="is unset"):
        validate_url_against_env("REDIS_URL", "redis")


def test_blank_env_refuses(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "  ")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    with pytest.raises(EnvMismatchError, match="is unset"):
        validate_url_against_env("REDIS_URL", "redis")


def test_invalid_env_value_refuses(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "staging")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    with pytest.raises(EnvMismatchError, match="must be 'local' or 'prod'"):
        validate_url_against_env("REDIS_URL", "redis")


def test_unset_url_refuses(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "local")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(EnvMismatchError, match="REDIS_URL is unset"):
        validate_url_against_env("REDIS_URL", "redis")


def test_prod_empty_host_refused_when_strict(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "prod")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@/db?host=/cloudsql/x")
    with pytest.raises(EnvMismatchError, match="has no host"):
        validate_url_against_env("DATABASE_URL", "postgres", allow_empty_host_in_prod=False)


# ---------- multi-URL helper ----------


def test_validate_all_or_raise_passes_with_local(monkeypatch):
    monkeypatch.setenv("ALMA_ENV", "local")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@postgres:5432/db")
    validate_all_or_raise(("REDIS_URL", "redis"), ("DATABASE_URL", "postgres"))


def test_validate_all_or_raise_first_mismatch_wins(monkeypatch):
    """If REDIS_URL is bad, raise on it — don't even look at DATABASE_URL."""
    monkeypatch.setenv("ALMA_ENV", "local")
    monkeypatch.setenv("REDIS_URL", "redis://prod.example.com:6379")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@postgres:5432/db")
    with pytest.raises(EnvMismatchError, match="REDIS_URL"):
        validate_all_or_raise(("REDIS_URL", "redis"), ("DATABASE_URL", "postgres"))
