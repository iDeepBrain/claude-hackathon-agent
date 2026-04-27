"""LLM provider factory with failover discovery.

The agent prefers Anthropic Claude (hackathon target) and falls back to
Gemini if Anthropic is unreachable or out of credits. The first
provider/model that responds to a ping is "pinned" for the rest of the
process — so the whole flow uses one coherent (provider, model) pair.

Reviewers can drop their own ANTHROPIC_API_KEY in .env and the agent will
automatically run on Claude Opus 4.7 with no code changes.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Provider chain built from .env at import time. The first (provider, model)
# pair that pings successfully gets pinned for the rest of the process.
#
# Order:
#   1. PRIMARY  HIGH (e.g. claude-opus-4-7)     — quality leader
#   2. PRIMARY  LOW  (e.g. claude-haiku-4-5)    — cheaper fallback same provider
#   3. SECONDARY HIGH (e.g. gemini-2.5-pro)     — backup provider
#   4. SECONDARY LOW (e.g. gemini-2.5-flash-lite) — cheapest backup
#
# Configure all four in .env to swap providers/models without code changes.

_VALID_PROVIDERS = {"anthropic", "gemini"}


def _build_chain() -> list[tuple[str, str, int]]:
    primary_provider = os.environ.get("LLM_PRIMARY_PROVIDER", "anthropic").strip().lower()
    primary_high    = os.environ.get("LLM_PRIMARY_MODEL_HIGH", "claude-opus-4-7").strip()
    primary_low     = os.environ.get("LLM_PRIMARY_MODEL_LOW",  "claude-haiku-4-5-20251001").strip()

    secondary_provider = os.environ.get("LLM_SECONDARY_PROVIDER", "gemini").strip().lower()
    secondary_high    = os.environ.get("LLM_SECONDARY_MODEL_HIGH", "gemini-2.5-pro").strip()
    secondary_low     = os.environ.get("LLM_SECONDARY_MODEL_LOW",  "gemini-2.5-flash-lite").strip()

    max_high = int(os.environ.get("LLM_MAX_TOKENS_HIGH", "2048"))
    max_low  = int(os.environ.get("LLM_MAX_TOKENS_LOW",  "1024"))

    chain: list[tuple[str, str, int]] = []
    for provider, model, tokens in [
        (primary_provider,   primary_high, max_high),
        (primary_provider,   primary_low,  max_low),
        (secondary_provider, secondary_high, max_high),
        (secondary_provider, secondary_low,  max_low),
    ]:
        if provider not in _VALID_PROVIDERS:
            logger.warning("Skipping unknown provider %r (must be 'anthropic' or 'gemini')", provider)
            continue
        if not model:
            continue
        chain.append((provider, model, tokens))

    if not chain:
        logger.warning("No valid LLM configured in .env, falling back to defaults")
        chain = [
            ("anthropic", "claude-opus-4-7",            2048),
            ("anthropic", "claude-haiku-4-5-20251001",  1024),
            ("gemini",    "gemini-2.5-pro",             2048),
            ("gemini",    "gemini-2.5-flash-lite",      1024),
        ]

    logger.info("LLM chain order:\n  %s", "\n  ".join(f"{p}/{m}" for p, m, _ in chain))
    return chain


PROVIDER_CHAIN: list[tuple[str, str, int]] = _build_chain()
PREFERRED_MODEL: str = PROVIDER_CHAIN[0][1] if PROVIDER_CHAIN else "claude-opus-4-7"

# Mutable globals — set by discover_provider() at startup.
_provider: str = "anthropic"
_default_model: str = PREFERRED_MODEL
_default_max_tokens: int = 2048
_fallback_reason: Optional[str] = None


# ── public getters / legacy setter ──────────────────────────────────────────

def get_provider() -> str:
    return _provider


def get_default_model() -> str:
    return _default_model


def get_preferred_model() -> str:
    return PREFERRED_MODEL


def get_fallback_reason() -> Optional[str]:
    return _fallback_reason


def set_provider(provider: str) -> None:
    """Legacy setter — still used by tests. Updates only the provider flag."""
    global _provider
    _provider = provider
    logger.info("LLM provider → %s", provider)


# ── pings (cheap, ≤ 12s timeout) ────────────────────────────────────────────

async def _ping_anthropic(model: str) -> tuple[bool, Optional[str]]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set"
    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            ),
            timeout=12.0,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:140]}"


async def _ping_gemini(model: str) -> tuple[bool, Optional[str]]:
    if not os.environ.get("GOOGLE_API_KEY"):
        return False, "GOOGLE_API_KEY not set"
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, max_output_tokens=5)
        await asyncio.wait_for(llm.ainvoke("ping"), timeout=12.0)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:140]}"


# ── discovery ──────────────────────────────────────────────────────────────

async def discover_provider() -> dict:
    """Probe each (provider, model) in PROVIDER_CHAIN. Pin the first that works.

    Returns a dict suitable for /health: {provider, model, preferred,
    fallback_reason}.
    """
    global _provider, _default_model, _default_max_tokens, _fallback_reason

    first_failure: Optional[str] = None
    for idx, (provider, model, max_tokens) in enumerate(PROVIDER_CHAIN):
        if provider == "anthropic":
            ok, err = await _ping_anthropic(model)
        else:
            ok, err = await _ping_gemini(model)

        if ok:
            _provider = provider
            _default_model = model
            _default_max_tokens = max_tokens
            _fallback_reason = first_failure if idx > 0 else None
            logger.info(
                "LLM elected: %s / %s%s",
                provider, model,
                f" (fallback from preferred — {first_failure})" if _fallback_reason else "",
            )
            return {
                "provider": provider,
                "model": model,
                "preferred": PREFERRED_MODEL,
                "fallback_reason": _fallback_reason,
            }

        logger.warning("LLM probe failed: %s/%s — %s", provider, model, err)
        if first_failure is None:
            first_failure = err

    raise RuntimeError(
        f"No LLM provider available. First error: {first_failure}"
    )


# ── factory used everywhere ─────────────────────────────────────────────────

def make_llm(model: str, max_tokens: int) -> BaseChatModel:
    """Build a chat model. Signature unchanged for backward compatibility.

    Behavior:
    - provider=anthropic + Opus discovered → respects `model` (router still
      drives Haiku/Sonnet/Opus selection)
    - provider=anthropic + only cheaper tier discovered → forces _default_model
      (avoids 400 if a higher tier is requested but no credits)
    - provider=gemini → ignores `model`, uses _default_model (already validated)
    """
    if _provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_default_model,
            max_output_tokens=max_tokens,
        )

    # provider == "anthropic"
    from langchain_anthropic import ChatAnthropic
    use_model = model if _default_model == PREFERRED_MODEL else _default_model
    return ChatAnthropic(model=use_model, max_tokens=max_tokens, streaming=True)


def build_image_content(text: str, image_b64: str) -> list:
    if _provider == "gemini":
        return [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
        {"type": "text", "text": text},
    ]
