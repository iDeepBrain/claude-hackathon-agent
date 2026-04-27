"""
Parametrized tests driven by CSV fixtures in tests/data/.
Each row is one test case; the language is determined by the file name (es/en).
"""
import csv
from pathlib import Path

import pytest

from app.agent.guard import is_injection, safe_response
from app.agent.persona import build_system_prompt, load_persona

_DATA_DIR = Path(__file__).parent / "data"


def _load_csv(filename: str) -> list[dict]:
    path = _DATA_DIR / filename
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows(filename: str, category: str | None = None) -> list[tuple]:
    rows = _load_csv(filename)
    if category:
        rows = [r for r in rows if r["category"] == category]
    return [(r["message"], r["expected_blocked"] == "true", r["description"]) for r in rows]


# ── Guard: injection detection ─────────────────────────────────────────────────

@pytest.mark.parametrize("message,expected_blocked,description", _rows("es.csv", "injection"))
def test_guard_injection_es(message, expected_blocked, description):
    blocked, _ = is_injection(message)
    assert blocked == expected_blocked, f"[ES] {description!r}: expected blocked={expected_blocked}"


@pytest.mark.parametrize("message,expected_blocked,description", _rows("en.csv", "injection"))
def test_guard_injection_en(message, expected_blocked, description):
    blocked, _ = is_injection(message)
    assert blocked == expected_blocked, f"[EN] {description!r}: expected blocked={expected_blocked}"


# ── Guard: normal messages pass through ───────────────────────────────────────

@pytest.mark.parametrize("message,expected_blocked,description", _rows("es.csv", "normal"))
def test_guard_normal_es(message, expected_blocked, description):
    blocked, pattern = is_injection(message)
    assert blocked == expected_blocked, (
        f"[ES] {description!r}: normal message was incorrectly blocked (pattern={pattern!r})"
    )


@pytest.mark.parametrize("message,expected_blocked,description", _rows("en.csv", "normal"))
def test_guard_normal_en(message, expected_blocked, description):
    blocked, pattern = is_injection(message)
    assert blocked == expected_blocked, (
        f"[EN] {description!r}: normal message was incorrectly blocked (pattern={pattern!r})"
    )


# ── safe_response language matches the request language ───────────────────────

@pytest.mark.parametrize("message,expected_blocked,description", _rows("es.csv", "injection"))
def test_safe_response_language_es(message, expected_blocked, description):
    if expected_blocked:
        resp = safe_response("es")
        assert any(word in resp.lower() for word in ["lo siento", "puedes", "mensaje"]), (
            f"[ES] safe_response should be in Spanish: {resp!r}"
        )


@pytest.mark.parametrize("message,expected_blocked,description", _rows("en.csv", "injection"))
def test_safe_response_language_en(message, expected_blocked, description):
    if expected_blocked:
        resp = safe_response("en")
        assert any(word in resp.lower() for word in ["sorry", "message", "try"]), (
            f"[EN] safe_response should be in English: {resp!r}"
        )


# ── Persona: system prompt contains language-appropriate content ───────────────

@pytest.mark.parametrize("message,expected_blocked,description", _rows("es.csv", "normal"))
def test_persona_es_prompt_used_for_es_messages(message, expected_blocked, description):
    prompt = build_system_prompt("", language="es")
    assert "Eres Alma" in prompt, "[ES] Spanish persona should start with 'Eres Alma'"


@pytest.mark.parametrize("message,expected_blocked,description", _rows("en.csv", "normal"))
def test_persona_en_prompt_used_for_en_messages(message, expected_blocked, description):
    prompt = build_system_prompt("", language="en")
    assert "You are Alma" in prompt, "[EN] English persona should start with 'You are Alma'"
