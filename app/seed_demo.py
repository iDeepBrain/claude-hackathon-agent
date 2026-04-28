"""Demo seed — populate the four memory layers for the canonical demo persona.

The demo persona is "Mateo" — a 28-year-old remote worker in Lima dealing with
chronic anxiety, irregular sleep, and an upcoming medical appointment. The
goal is to make portfolio reviewers see populated memory state on first
interaction, not empty schema fields.

This module is invoked from two places:
    1. The /api/v1/demo/seed endpoint (production reset for live demos)
    2. The scripts/reset_demo.py CLI (local dev)

Idempotent: calling seed_mateo twice produces the same final state because
upsert_memory_tool merges by record identity.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEMO_USER_ID_DEFAULT = "demo_mateo"


def _next_friday_iso() -> str:
    today = datetime.now(timezone.utc).date()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


def _build_seed_data() -> dict[str, list[dict[str, Any]]]:
    today = datetime.now(timezone.utc).date()
    next_friday = _next_friday_iso()

    mood_history = [
        {"date": (today - timedelta(days=7)).isoformat(), "mood": "tristeza_baja", "intensity": 5, "context": "día normal sin energía"},
        {"date": (today - timedelta(days=6)).isoformat(), "mood": "ansiedad_moderada", "intensity": 6, "context": "el jefe mencionó deadline ajustado"},
        {"date": (today - timedelta(days=5)).isoformat(), "mood": "ansiedad_moderada", "intensity": 7, "context": "no durmió bien, despertó 3am"},
        {"date": (today - timedelta(days=4)).isoformat(), "mood": "neutral", "intensity": 5, "context": "domingo tranquilo en familia"},
        {"date": (today - timedelta(days=3)).isoformat(), "mood": "ansiedad_alta", "intensity": 7, "context": "reunión con jefe sobre rendimiento"},
        {"date": (today - timedelta(days=2)).isoformat(), "mood": "estrés", "intensity": 7, "context": "insomnio nuevamente"},
        {"date": (today - timedelta(days=1)).isoformat(), "mood": "ansiedad_moderada", "intensity": 6, "context": "anticipando cita médica del viernes"},
    ]

    mentioned_events = [
        {"date": next_friday, "event": "Cita médica — chequeo general", "category": "salud"},
        {"date": (today - timedelta(days=3)).isoformat(), "event": "Reunión con jefe sobre rendimiento", "category": "trabajo"},
        {"date": (today + timedelta(days=2)).isoformat(), "event": "Visita de su hermana desde Trujillo", "category": "familia"},
        {"date": "2026-02-01", "event": "Cambio a trabajo remoto permanente", "category": "trabajo"},
    ]

    habits = [
        {"habit": "Sueño irregular", "detail": "Frecuentemente despierta entre 3-4am sin poder volver a dormir", "since": "2026-02"},
        {"habit": "Café como compensación", "detail": "3-4 tazas matinales para compensar fatiga", "since": "2026-02"},
        {"habit": "Sedentarismo", "detail": "Sin ejercicio regular desde el inicio del trabajo remoto", "since": "2026-02"},
    ]

    interaction_prefs = [
        {"preference": "language", "value": "es", "note": "Spanish, Lima dialect"},
        {"preference": "response_length", "value": "concise", "note": "Le incomodan respuestas largas o muy terapéuticas"},
        {"preference": "tone", "value": "warm-direct", "note": "Aprecia preguntas concretas sobre el día"},
        {"preference": "timezone", "value": "America/Lima", "note": "UTC-5"},
    ]

    return {
        "mood_history": mood_history,
        "mentioned_events": mentioned_events,
        "habits": habits,
        "interaction_prefs": interaction_prefs,
    }


async def seed_mateo(mcp_client: Any, user_id: str = DEMO_USER_ID_DEFAULT) -> dict[str, int]:
    """Populate the four memory layers for the demo user via MCP.

    Returns {layer: count_written} for verification. Failures are logged
    but do not raise — partial seeds are acceptable for a demo helper.
    """
    seed_data = _build_seed_data()
    counts: dict[str, int] = {}

    for layer, records in seed_data.items():
        written = 0
        for record in records:
            result = await mcp_client.upsert_memory(user_id, layer, record)
            if result.get("success", True) is not False:
                written += 1
            else:
                logger.warning("Seed upsert failed: layer=%s record=%s", layer, record)
        counts[layer] = written

    logger.info("Seeded demo user %r — counts=%s", user_id, counts)
    return counts
