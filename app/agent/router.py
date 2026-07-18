"""Per-message model routing.

Picks (model, max_tokens) for each turn. Respects the provider that
discover_provider() actually pinned at startup — if Anthropic is out of
credits and the agent fell back to Gemini, the router uses the active
Gemini model instead of insisting on Claude (which would crash the
stream silently — see WS-H.6 incident notes).

Token budgets come from env vars (LLM_MAX_TOKENS_HIGH / LOW) so an
operator can bump them without redeploying the image — useful when
demos show responses cut mid-sentence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Hardcoded fallbacks ONLY used if discovery hasn't run yet (e.g. unit
# tests that import this module without lifespan). Production always
# overrides via app.agent.llm._default_model after discover_provider().
_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"
_OPUS = "claude-opus-4-7"


@dataclass
class ModelConfig:
    model: str
    max_tokens: int = 2048


def _max_tokens_high() -> int:
    return int(os.environ.get("LLM_MAX_TOKENS_HIGH", "4096"))


def _max_tokens_low() -> int:
    return int(os.environ.get("LLM_MAX_TOKENS_LOW", "2048"))


def _active_model_low() -> str:
    """Default ("low") model for routine turns.

    Prefers the model discover_provider() pinned at startup so the router
    never sends a request to a provider that's already known to be down.
    Falls back to Haiku only when imported in test contexts.
    """
    try:
        from app.agent.llm import get_default_model
        active = get_default_model()
        if active:
            return active
    except Exception:
        pass
    return _HAIKU


def _active_model_high() -> str:
    """Higher-tier model for crisis / long-message turns.

    For Anthropic primary, this is Opus. For a Gemini fallback, the
    LLM module already pinned a single model, so we reuse it — there's
    no point asking for a "stronger" Gemini if the primary already
    failed to ping.
    """
    try:
        from app.agent.llm import get_default_model, get_provider
        if get_provider() == "anthropic":
            # Anthropic family — bump to Opus for high-stakes turns
            return os.environ.get("LLM_PRIMARY_MODEL_HIGH", _OPUS)
        # Gemini family — reuse whatever was pinned (flash-lite typically)
        active = get_default_model()
        if active:
            return active
    except Exception:
        pass
    return _OPUS


def route_model(session_state: str, message: str, has_image: bool = False) -> ModelConfig:
    if session_state == "crisis":
        return ModelConfig(model=_active_model_high(), max_tokens=_max_tokens_high())
    if has_image:
        return ModelConfig(model=_active_model_low(), max_tokens=_max_tokens_low())
    if len(message) > 800:
        return ModelConfig(model=_active_model_high(), max_tokens=_max_tokens_high())
    return ModelConfig(model=_active_model_low(), max_tokens=_max_tokens_low())
