import pytest
from app.agent.persona import build_system_prompt, load_persona


def test_load_persona_spanish():
    text = load_persona("es")
    assert isinstance(text, str)
    assert len(text) > 100
    # Spanish prompt written in Spanish
    assert "Eres Alma" in text


def test_load_persona_english():
    text = load_persona("en")
    assert isinstance(text, str)
    assert len(text) > 100
    # English prompt written in English
    assert "You are Alma" in text


def test_load_persona_unknown_language_falls_back_to_spanish():
    text = load_persona("fr")
    assert text == load_persona("es")


def test_load_persona_default_is_spanish():
    assert load_persona() == load_persona("es")


def test_es_and_en_prompts_are_different():
    assert load_persona("es") != load_persona("en")


# ── build_system_prompt ────────────────────────────────────────────────────────

def test_build_system_prompt_no_context_returns_persona_only():
    result = build_system_prompt("", language="es")
    assert result == load_persona("es")


def test_build_system_prompt_empty_history_sentinel_returns_persona_only():
    result = build_system_prompt("Sin historial previo para este usuario.", language="es")
    assert result == load_persona("es")


def test_build_system_prompt_with_context_es():
    ctx = "Nombre: María\nEstado emocional: 6/10"
    result = build_system_prompt(ctx, language="es")
    assert "Contexto actual del usuario" in result
    assert ctx in result
    assert load_persona("es") in result


def test_build_system_prompt_with_context_en():
    ctx = "Name: John\nEmotional state: 7/10"
    result = build_system_prompt(ctx, language="en")
    assert "Current user context" in result
    assert ctx in result
    assert load_persona("en") in result


def test_build_system_prompt_context_header_matches_language_es():
    result = build_system_prompt("algo de contexto", language="es")
    assert "Contexto actual del usuario" in result
    assert "Current user context" not in result


def test_build_system_prompt_context_header_matches_language_en():
    result = build_system_prompt("some context", language="en")
    assert "Current user context" in result
    assert "Contexto actual del usuario" not in result


def test_build_system_prompt_default_language_is_es():
    ctx = "datos del usuario"
    assert build_system_prompt(ctx) == build_system_prompt(ctx, language="es")
