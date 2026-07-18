"""Tests for app.agent.router after WS-H.6.

The router now respects the model that discover_provider() pinned at
startup (so a Gemini fallback doesn't get sent Claude requests) and
reads max_tokens from env vars (so an operator can bump them without
redeploying the image).
"""
import pytest

from app.agent.router import ModelConfig, route_model


@pytest.fixture
def claude_active(monkeypatch):
    """Mock LLM module state to claim Claude is the pinned provider."""
    monkeypatch.setattr("app.agent.llm.get_provider", lambda: "anthropic")
    monkeypatch.setattr("app.agent.llm.get_default_model", lambda: "claude-haiku-4-5-20251001")
    monkeypatch.setenv("LLM_PRIMARY_MODEL_HIGH", "claude-opus-4-7")
    monkeypatch.setenv("LLM_MAX_TOKENS_HIGH", "4096")
    monkeypatch.setenv("LLM_MAX_TOKENS_LOW", "2048")


@pytest.fixture
def gemini_active(monkeypatch):
    """Mock state for the post-fallback case (Anthropic out of credits)."""
    monkeypatch.setattr("app.agent.llm.get_provider", lambda: "gemini")
    monkeypatch.setattr("app.agent.llm.get_default_model", lambda: "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_MAX_TOKENS_HIGH", "4096")
    monkeypatch.setenv("LLM_MAX_TOKENS_LOW", "2048")


# ── Anthropic primary path ──────────────────────────────────────────────────


def test_default_uses_active_low_model_anthropic(claude_active):
    cfg = route_model("chat", "hola")
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 2048


def test_crisis_uses_high_model_anthropic(claude_active):
    cfg = route_model("crisis", "estoy bien")
    assert cfg.model == "claude-opus-4-7"
    assert cfg.max_tokens == 4096


def test_long_message_uses_high_model_anthropic(claude_active):
    cfg = route_model("chat", "x" * 801)
    assert cfg.model == "claude-opus-4-7"
    assert cfg.max_tokens == 4096


def test_image_uses_low_model_anthropic(claude_active):
    cfg = route_model("chat", "mira esta foto", has_image=True)
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 2048


def test_crisis_overrides_image_anthropic(claude_active):
    cfg = route_model("crisis", "imagen", has_image=True)
    assert cfg.model == "claude-opus-4-7"
    assert cfg.max_tokens == 4096


# ── Gemini fallback path (the bug WS-H.6 fixed) ─────────────────────────────


def test_default_uses_active_low_model_gemini(gemini_active):
    """When discovery falls back to Gemini, the router must NOT send Claude."""
    cfg = route_model("chat", "hola")
    assert cfg.model == "gemini-2.5-flash-lite"
    assert "claude" not in cfg.model


def test_crisis_reuses_pinned_gemini_model(gemini_active):
    """No point asking for Opus when only flash-lite is reachable."""
    cfg = route_model("crisis", "estoy mal")
    assert cfg.model == "gemini-2.5-flash-lite"
    assert cfg.max_tokens == 4096


def test_long_message_reuses_pinned_gemini(gemini_active):
    cfg = route_model("chat", "x" * 801)
    assert cfg.model == "gemini-2.5-flash-lite"
    assert cfg.max_tokens == 4096


# ── env-driven max_tokens ───────────────────────────────────────────────────


def test_max_tokens_high_respects_env(claude_active, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS_HIGH", "8192")
    cfg = route_model("crisis", "x")
    assert cfg.max_tokens == 8192


def test_max_tokens_low_respects_env(claude_active, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS_LOW", "3000")
    cfg = route_model("chat", "hola")
    assert cfg.max_tokens == 3000


def test_max_tokens_default_when_env_missing(claude_active, monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS_HIGH", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS_LOW", raising=False)
    cfg_low = route_model("chat", "hola")
    cfg_high = route_model("crisis", "x")
    assert cfg_low.max_tokens == 2048
    assert cfg_high.max_tokens == 4096


# ── ModelConfig contract ────────────────────────────────────────────────────


def test_model_config_is_dataclass(claude_active):
    cfg = route_model("onboarding", "hola")
    assert isinstance(cfg, ModelConfig)
    assert isinstance(cfg.max_tokens, int)
    assert cfg.max_tokens > 0
