import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import build_image_content, make_llm

from app.agent.guard import (
    fast_crisis_precheck,
    is_injection,
    is_meta_query,
    is_off_topic,
    is_persona_drift,
    looks_like_code_output,
    looks_like_json_output,
    looks_like_persona_leak,
    looks_like_technical_output,
    meta_query_response,
    off_topic_response,
    safe_response,
)
from app.agent.persona import build_system_prompt
from app.agent.router import route_model
from app.cache.semantic import SemanticCache
from app.cache.session import RedisSession
from app.mcp_client.client import MCPClient

logger = logging.getLogger(__name__)

_SEARCH_SCORE_MIN = 0.5  # minimum cosine similarity to include a search result
_SEARCH_K = 3

_EVENT_PHRASES = [
    "mañana",
    "el lunes",
    "el martes",
    "el miércoles",
    "el jueves",
    "el viernes",
    "la semana que viene",
    "la semana pasada",
    "mi trabajo",
    "mi jefe",
    "mi pareja",
    "mi familia",
    "mi mamá",
    "mi papá",
    "tomorrow",
    "next week",
    "last week",
    "my job",
    "my partner",
    "my boss",
]

_CRISIS_THRESHOLD = 0.4

# mood_history upserts ONE entry per day (key "mood_<YYYY-MM-DD>"), so the
# threshold is also "minimum days of usage." Set to 3 because a single bar
# (or even two) doesn't communicate a "week" — it's noise in the panel
# and confuses anonymous demo users on day one. The timeline appears
# meaningfully when there's actual multi-day signal — which in practice
# means after the user logs in (Google OAuth in WS-D.1) and their
# accumulated data is fetched, OR after 3+ days of organic usage on the
# same anonymous UUID.
_MOOD_SUMMARY_MIN_ENTRIES = 3
_MOOD_SUMMARY_DAYS = 7
_MEMORY_CARD_MIN_SCORE = 0.6  # below this, the recall isn't confident enough to surface
_MEMORY_CARD_MAX_CHARS = 280  # cap chunk length so a single big record doesn't dominate UI

# Crisis alert tiers (calibrated against the MCP deterministic detector):
#   >= 0.1 = single MEDIUM keyword ("muy triste", "deprimido", "agotado"...)
#           — surface the alert so the user/reviewer sees the safety layer
#           is reasoning about this turn, not pretending nothing happened
#   >= 0.2 = two MEDIUM matches OR strong combinations
#   >= 0.4 = HARD keyword ("no quiero vivir", "kill myself"...) — chain ALSO
#           transitions session.state to "crisis" via the existing _post_response
#   >= 0.6 = proactive Redis gate active — Cloud Scheduler suppresses outbound
_CRISIS_ALERT_MIN_SCORE = 0.1
_CRISIS_PROACTIVE_GATE = 0.6


def _format_memory_chunk(content: object) -> str:
    """Format a memory record as a compact human-readable single line.

    Records are stored as dicts (e.g. ``{date, event, category}`` for
    mentioned_events, ``{description, message, response_preview}`` for
    auto-saved exchanges). Joining the truthy values with `` · `` gives a
    label-like line, but naive joining duplicates content when one field
    is a truncated copy of another (``description = message[:120]``). The
    dedup keeps only the LONGEST value for each substring chain.

    Falsy values (None/""/False) are dropped. The final line is capped at
    ``_MEMORY_CARD_MAX_CHARS`` so a single record can't dominate the UI.
    """
    if isinstance(content, dict):
        raw = [str(v).strip() for v in content.values() if v not in (None, "", False)]
        raw = [v for v in raw if v]
        # Sort longest-first so substring drops are deterministic
        ordered = sorted(raw, key=len, reverse=True)
        kept: list[str] = []
        for v in ordered:
            # Skip if a longer already-kept value already contains this one
            if any(v in k for k in kept):
                continue
            kept.append(v)
        text = " · ".join(kept)
    else:
        text = str(content) if content is not None else ""
    # Cycle 2 finding: long digit/decimal runs (JS Date.now() leaks like
    # "1777413404695.0568372488704...") survived the prior 8+ digit
    # regex when they had decimal points. Strip aggressively here so
    # both the inline render_memory_card payload AND any other surface
    # that re-renders this string sees a clean version.
    text = re.sub(r"\b\d{6,}(?:[.,]\d+)?\b", "", text)
    text = re.sub(r"\s+·\s*·\s+", " · ", text)  # collapse adjacent dot separators
    text = re.sub(r"\s+", " ", text).strip(" ·")
    if len(text) > _MEMORY_CARD_MAX_CHARS:
        text = text[: _MEMORY_CARD_MAX_CHARS - 1] + "…"
    return text


def _extract_week_summary(mood_history: list) -> list[dict]:
    """Extract the last 7 days of mood entries in a render-friendly shape.

    Tolerates both the production schema (``mood_score`` + ``entry_key``
    formatted ``mood_YYYY-MM-DD``) and the seed schema (``date`` + ``intensity``).
    Tolerates wrapper shapes where MCP returns ``{content, ts, ...}``.
    """
    week: list[dict] = []
    for entry in mood_history[-_MOOD_SUMMARY_DAYS:]:
        # Memory MCP may wrap records as {"content": {...}, ...}
        record = entry.get("content", entry) if isinstance(entry, dict) else {}
        if not isinstance(record, dict):
            continue

        # Date: either explicit field or parsed from entry_key like "mood_2026-04-22"
        d = record.get("date")
        if not d and isinstance(record.get("entry_key"), str):
            d = record["entry_key"].replace("mood_", "") or None

        # Score: 0-10 in production (mood_score) or 1-10 in seed (intensity)
        score = record.get("mood_score")
        if score is None:
            score = record.get("intensity")
        if score is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        week.append({
            "date": d or "",
            "score": round(score, 1),
            "context": str(record.get("context") or record.get("mood") or ""),
        })
    return week


def _build_human_content(text: str, image_b64: str | None) -> list | str:
    if image_b64 is None:
        return text
    return build_image_content(text, image_b64)


def _has_event_mention(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in _EVENT_PHRASES)


def _extract_event_snippet(message: str) -> str | None:
    """Extract a clean event description from a free-text user message.

    Replaces the previous `description=message[:120]` auto-upsert pattern
    which stored the user's raw message verbatim — that polluted the
    themes panel with timestamps and full sentences. This extracts the
    matched event phrase with a small context window and strips numeric
    timestamps that are clearly testing artifacts (8+ contiguous digits).

    Returns ``None`` if no event phrase matches, otherwise a string
    capped at 80 chars with whitespace normalized.
    """
    lower = message.lower()
    # Find the earliest matched phrase
    earliest: tuple[str, int] | None = None
    for phrase in _EVENT_PHRASES:
        idx = lower.find(phrase)
        if idx >= 0 and (earliest is None or idx < earliest[1]):
            earliest = (phrase, idx)
    if not earliest:
        return None
    phrase, idx = earliest
    # Start at the matched phrase (don't take preamble — risks mid-word cut).
    end = min(len(message), idx + len(phrase) + 50)
    snippet = message[idx:end]
    # Strip long digit sequences (timestamps, IDs from testing pollution)
    snippet = re.sub(r"\b\d{8,}\b", "", snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:80] if snippet else None


class AlmaChain:
    def __init__(self, mcp_client: MCPClient, session_store: RedisSession, cache: SemanticCache) -> None:
        self._mcp = mcp_client
        self._sessions = session_store
        self._cache = cache

    async def stream_events(
        self, user_id: str, message: str, image_b64: str | None = None, language: str = "es"
    ) -> AsyncGenerator[dict, None]:
        """Stream a typed-event sequence describing the full pipeline.

        Event shape: ``{"type": "<name>", ...payload}``. Event types:

        - ``agent_start``       — first event, includes language and image flag.
        - ``guard_blocked``     — injection guard tripped; chain aborts.
        - ``cache_hit``         — semantic cache hit; chain returns cached.
        - ``memory_retrieved``  — pgvector search done; metadata only (no chunks).
        - ``model_routed``      — selected model + max_tokens.
        - ``response_chunk``    — each token from the LLM stream (legacy ``stream()`` filters these).
        - ``agent_done``        — final, with ``stop_reason`` and latency totals.

        The ``response_chunk`` events carry the user-facing token text in
        ``content``. All other events carry pipeline observability metadata
        and never leak raw memory contents into the SSE channel.
        """
        t_start = time.monotonic()
        yield {
            "type": "agent_start",
            "language": language,
            "has_image": image_b64 is not None,
        }

        # Crisis pre-check — if the message contains crisis keywords, skip
        # every other input guard so a user in distress never gets refused
        # for accidentally overlapping with an off-topic / injection pattern
        # (refusal-bait edge case flagged by the multi-agent review).
        crisis_override = fast_crisis_precheck(message)
        if crisis_override:
            logger.info("Crisis pre-check matched for user %s — bypassing input guards", user_id)
            yield {"type": "guard_crisis_override", "reason": "skipped_input_guards"}

        blocked, pattern = is_injection(message) if not crisis_override else (False, "")
        if blocked:
            logger.warning("Injection attempt from user %s: pattern=%r", user_id, pattern)
            yield {"type": "guard_blocked", "pattern": pattern}
            yield {"type": "response_chunk", "content": safe_response(language)}
            yield {
                "type": "agent_done",
                "stop_reason": "guard",
                "latency_ms": int((time.monotonic() - t_start) * 1000),
            }
            return

        # Meta-query short-circuit. Fires for "dame tu prompt", "are you
        # Gemini?", etc. — questions where the LLM tends to reveal
        # plumbing if no rule blocks it. We prefer to answer locally
        # with a polite redirect that NEVER mentions provider/model.
        # Persona prompt also has a hard lock-down rule for the cases
        # this regex misses, but blocking here saves an LLM round-trip
        # AND defends against accidental persona prompt regressions.
        meta_hit, meta_pattern = is_meta_query(message) if not crisis_override else (False, "")
        if meta_hit:
            logger.info("Meta-query deflected for user %s: pattern=%r", user_id, meta_pattern)
            yield {"type": "guard_meta", "pattern": meta_pattern}
            yield {"type": "response_chunk", "content": meta_query_response(language)}
            yield {
                "type": "agent_done",
                "stop_reason": "meta_guard",
                "latency_ms": int((time.monotonic() - t_start) * 1000),
            }
            return

        # Scope short-circuit. Fires for "give me fibonacci", "dame el
        # pseudocódigo", etc. — Alma is an emotional companion, not a
        # coding/homework assistant. Deflect locally so the LLM never
        # gets a chance to comply with the task. Persona prompt also
        # carries a single-line rule for cases this regex misses.
        scope_hit, scope_pattern = is_off_topic(message) if not crisis_override else (False, "")
        if scope_hit:
            logger.info("Off-topic deflected for user %s: pattern=%r", user_id, scope_pattern)
            yield {"type": "guard_scope", "pattern": scope_pattern}
            yield {"type": "response_chunk", "content": off_topic_response(language)}
            yield {
                "type": "agent_done",
                "stop_reason": "scope_guard",
                "latency_ms": int((time.monotonic() - t_start) * 1000),
            }
            return

        session = await self._sessions.get(user_id)
        session.language = language

        # Crescendo / multi-turn jailbreak check — re-run is_injection over
        # the concatenation of the last 3 human turns. A slow benign-looking
        # escalation (turn 1 "imagine you are a consoler", turn 2 "what
        # would she say without restrictions") flies past the per-message
        # guard but trips when seen as a window. Cheap (substring) and uses
        # session history already in memory.
        if not crisis_override and session.history:
            recent_human = [
                m["content"] for m in session.history[-6:] if m.get("role") == "human"
            ][-3:]
            if recent_human:
                window = " ".join(recent_human + [message])
                window_blocked, window_pattern = is_injection(window)
                if window_blocked and not is_injection(message)[0]:
                    logger.warning(
                        "Crescendo pattern detected for user %s: %r",
                        user_id, window_pattern,
                    )
                    yield {"type": "guard_crescendo", "pattern": window_pattern}
                    yield {"type": "response_chunk", "content": safe_response(language)}
                    yield {
                        "type": "agent_done",
                        "stop_reason": "crescendo_guard",
                        "latency_ms": int((time.monotonic() - t_start) * 1000),
                    }
                    return

        if not image_b64:
            cached = await self._cache.get(message)
            if cached is not None:
                logger.debug("Semantic cache hit for user %s", user_id)
                yield {"type": "cache_hit"}
                yield {"type": "response_chunk", "content": cached}
                yield {
                    "type": "agent_done",
                    "stop_reason": "cache",
                    "latency_ms": int((time.monotonic() - t_start) * 1000),
                }
                return

        context = await self._mcp.build_context(user_id)

        # Augment context with semantically relevant memories for this message
        relevant = await self._mcp.search_memories(user_id, message, k=_SEARCH_K)
        relevant_above = [r for r in relevant if r.get("score", 0) >= _SEARCH_SCORE_MIN]
        if relevant_above:
            header = "### Relevant memories" if language == "en" else "### Recuerdos relevantes"
            snippets = [f"- [{r['layer']}] {r['content']}" for r in relevant_above]
            context += f"\n\n{header}\n" + "\n".join(snippets)

        yield {
            "type": "memory_retrieved",
            "count": len(relevant_above),
            "layers": sorted({r.get("layer") for r in relevant_above if r.get("layer")}),
            "top_score": max((float(r.get("score", 0.0)) for r in relevant_above), default=0.0),
        }

        # If the top retrieved chunk is highly relevant, surface it as a
        # user-facing card so the user SEES what Alma recalled — verbatim
        # (anti-hallucination: the LLM doesn't paraphrase the memory; the
        # exact stored chunk is shown). This is a separate event from
        # memory_retrieved (which carries metadata only — see privacy test).
        if relevant_above:
            top = max(relevant_above, key=lambda r: float(r.get("score", 0.0)))
            top_score = float(top.get("score", 0.0))
            if top_score >= _MEMORY_CARD_MIN_SCORE:
                chunk_text = _format_memory_chunk(top.get("content"))
                if chunk_text:
                    yield {
                        "type": "render_memory_card",
                        "chunk": chunk_text,
                        "layer": str(top.get("layer", "")),
                        "score": round(top_score, 3),
                    }

        model_cfg = route_model(session.state, message, bool(image_b64))
        yield {
            "type": "model_routed",
            "model": model_cfg.model,
            "max_tokens": model_cfg.max_tokens,
            "session_state": session.state,
        }

        # Kick off mood-summary AND crisis-evaluation in parallel with the
        # LLM stream so the post-stream renders have data ready before tokens
        # finish. Both tasks are bounded by short timeouts on await so neither
        # ever blocks stream completion past a fraction of a second.
        mood_task = asyncio.create_task(self._mcp.get_memory(user_id))
        crisis_task = asyncio.create_task(self._mcp.evaluate_crisis_risk(user_id, message))

        system_prompt = build_system_prompt(context, language)
        llm = make_llm(model_cfg.model, model_cfg.max_tokens)

        history_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
            for m in session.history
        ]

        current_human = HumanMessage(content=_build_human_content(message, image_b64))

        messages = [SystemMessage(content=system_prompt)] + history_messages + [current_human]

        full_response_chunks: list[str] = []
        chunk_count = 0

        # Output guard — abort the stream if the model starts writing code
        # or pseudocode. Inspect the first ~120 chars of accumulated output;
        # once that buffer trips looks_like_code_output we stop forwarding
        # tokens, replace what's been emitted with a redirect message, and
        # close the stream. Cheap (substring matches), runs once per chunk
        # until the buffer crosses the inspection threshold.
        _OUTPUT_INSPECT_CHARS = 120
        output_aborted = False
        accumulated = ""
        inspected = False

        async for chunk in llm.astream(messages):
            text = chunk.content
            if isinstance(text, list):
                # Vision responses can have list content blocks
                text = "".join(block.get("text", "") for block in text if isinstance(block, dict))
            if not text:
                continue
            accumulated += text
            if not inspected and len(accumulated) >= _OUTPUT_INSPECT_CHARS:
                inspected = True
                if looks_like_code_output(accumulated):
                    logger.warning(
                        "Output code-shape detected for user %s — aborting stream", user_id
                    )
                    output_aborted = True
                    yield {"type": "guard_output", "reason": "code_shape"}
                    yield {"type": "response_chunk", "content": off_topic_response(language)}
                    break
                if looks_like_technical_output(accumulated):
                    logger.warning(
                        "Output technical-shape detected for user %s — aborting stream", user_id
                    )
                    output_aborted = True
                    yield {"type": "guard_output", "reason": "technical_shape"}
                    yield {"type": "response_chunk", "content": off_topic_response(language)}
                    break
                if looks_like_persona_leak(accumulated):
                    logger.warning(
                        "Persona-leak shape detected for user %s — aborting stream", user_id
                    )
                    output_aborted = True
                    yield {"type": "guard_output", "reason": "persona_leak"}
                    yield {"type": "response_chunk", "content": meta_query_response(language)}
                    break
            full_response_chunks.append(text)
            chunk_count += 1
            yield {"type": "response_chunk", "content": text}

        if output_aborted:
            full_response = off_topic_response(language)
        else:
            full_response = "".join(full_response_chunks)
            # Final-pass output guards — JSON dump (parser/SSE bug) or
            # persona drift (echo, empty, canned LLM artifact). Both are
            # bug shapes; replace with the safe response so the user never
            # sees raw payload or "As an AI…" fallthrough.
            if looks_like_json_output(full_response):
                logger.warning(
                    "Output JSON dump detected for user %s — replacing with safe response",
                    user_id,
                )
                yield {"type": "guard_output", "reason": "json_dump"}
                yield {"type": "response_chunk", "content": safe_response(language)}
                full_response = safe_response(language)
            elif is_persona_drift(full_response, message):
                logger.warning(
                    "Persona drift detected for user %s — replacing with safe response",
                    user_id,
                )
                yield {"type": "guard_output", "reason": "persona_drift"}
                yield {"type": "response_chunk", "content": safe_response(language)}
                full_response = safe_response(language)
        latency_ms = int((time.monotonic() - t_start) * 1000)

        # Render a small weekly mood timeline if we have enough history.
        # The fetch was kicked off in parallel with the LLM stream above, so
        # the await is typically a no-op. Bound the wait so a slow MCP call
        # never blocks stream completion past a fraction of a second.
        try:
            mem = await asyncio.wait_for(mood_task, timeout=0.5)
            mood_history = mem.get("mood_history", []) if isinstance(mem, dict) else []
            if len(mood_history) >= _MOOD_SUMMARY_MIN_ENTRIES:
                week = _extract_week_summary(mood_history)
                if week:
                    yield {
                        "type": "render_session_summary",
                        "week": week,
                    }
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("Skipping render_session_summary: %s", exc)

        # Surface the crisis evaluation as a visible artifact so the user
        # (and demo reviewers) SEE that the safety layer is reasoning about
        # this turn — not opaque magic. Only fired above MIN_SCORE so the
        # baseline "all good" doesn't burden the UI. Includes the proactive
        # gate state so it's clear what the system is DOING with the score.
        try:
            crisis = await asyncio.wait_for(crisis_task, timeout=0.5)
            score = float(crisis.get("score", 0.0)) if isinstance(crisis, dict) else 0.0
            if score >= _CRISIS_ALERT_MIN_SCORE:
                yield {
                    "type": "render_crisis_alert",
                    "score": round(score, 2),
                    "level": str(crisis.get("level", "low")) if isinstance(crisis, dict) else "low",
                    "gates": {
                        # gate #1 — chain-side, decided by score alone
                        "proactive_suppressed": score >= _CRISIS_PROACTIVE_GATE,
                    },
                }
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("Skipping render_crisis_alert: %s", exc)

        yield {
            "type": "agent_done",
            "stop_reason": "end_turn",
            "chunks": chunk_count,
            "response_chars": len(full_response),
            "latency_ms": latency_ms,
            # Tracks SystemMessage delivery integrity — if Gemini fallback
            # silently drops the system prompt, this number collapses and
            # the persona disappears. Watched in logs to catch adapter bugs.
            "system_prompt_len": len(system_prompt),
        }

        # Post-response work runs in background so the stream closes immediately
        asyncio.create_task(
            self._post_response(user_id, message, full_response, session, image_b64)
        )

    async def stream(
        self, user_id: str, message: str, image_b64: str | None = None, language: str = "es"
    ) -> AsyncGenerator[str, None]:
        """Legacy string-only stream API.

        Kept stable for existing callers and tests. Wraps ``stream_events()``
        and yields only the user-facing response text — pipeline observability
        events are dropped here. New consumers should use ``stream_events()``.
        """
        async for event in self.stream_events(user_id, message, image_b64, language):
            if event["type"] == "response_chunk":
                yield event["content"]

    async def _post_response(
        self,
        user_id: str,
        message: str,
        response: str,
        session,
        image_b64: str | None,
    ) -> None:
        # Update history (trim to 20 turns = 40 messages)
        session.history.append({"role": "human", "content": message})
        session.history.append({"role": "assistant", "content": response})
        session.history = self._sessions._trim_history(session.history, max_turns=40)

        # Crisis risk evaluation
        risk = await self._mcp.evaluate_crisis_risk(user_id, message)
        if risk.get("score", 0.0) > _CRISIS_THRESHOLD and session.state != "crisis":
            session.state = "crisis"
            logger.info("User %s entered crisis state (score=%.2f)", user_id, risk["score"])
        elif risk.get("score", 0.0) <= _CRISIS_THRESHOLD and session.state == "crisis":
            session.state = "chat"

        # Transition out of onboarding after first exchange
        if session.state == "onboarding":
            session.state = "chat"

        await self._sessions.save(user_id, session)

        await self._sessions._redis.set(
            f"alma:session:last_activity:{user_id}",
            str(datetime.now().timestamp()),
        )

        # Cache response for non-image messages
        if image_b64 is None:
            await self._cache.set(message, response)

        crisis_score = risk.get("score", 0.0)
        await self._sessions._redis.set(
            f"alma:crisis:last:{user_id}",
            str(crisis_score),
        )

        # Daily mood snapshot — upsert by date key so one entry per day
        await self._mcp.upsert_memory(
            user_id,
            "mood_history",
            {
                "entry_key": f"mood_{date.today().isoformat()}",
                "mood_score": round((1.0 - crisis_score) * 10, 1),
                "crisis_score": crisis_score,
                "crisis_level": risk.get("level", "low"),
                "session_state": session.state,
                "emotional_weight": crisis_score,
            },
        )

        # Memory upsert for mentioned events: store ONLY a clean, short
        # event field — not the raw user message. Previously this stored
        # description=message[:120] AND the full message AND a response
        # preview, which polluted the themes panel with timestamps and
        # full sentences. The cleaner format renders well in the panel
        # and the render_memory_card still gets richer context via the
        # surrounding response.
        snippet = _extract_event_snippet(message)
        if snippet:
            # Memory-poisoning guard — never persist a snippet that itself
            # carries injection / off-topic shape, otherwise it would be
            # injected into a future system prompt as "remembered context"
            # and re-prime the LLM against the persona.
            if is_injection(snippet)[0] or is_off_topic(snippet)[0]:
                logger.warning(
                    "Memory upsert blocked for user %s — snippet shape: %r",
                    user_id, snippet[:50],
                )
            else:
                await self._mcp.upsert_memory(
                    user_id,
                    "mentioned_events",
                    {"event": snippet},
                )
