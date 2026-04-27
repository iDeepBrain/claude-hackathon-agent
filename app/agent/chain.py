import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import build_image_content, make_llm

from app.agent.guard import is_injection, safe_response
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


def _build_human_content(text: str, image_b64: str | None) -> list | str:
    if image_b64 is None:
        return text
    return build_image_content(text, image_b64)


def _has_event_mention(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in _EVENT_PHRASES)


class AlmaChain:
    def __init__(self, mcp_client: MCPClient, session_store: RedisSession, cache: SemanticCache) -> None:
        self._mcp = mcp_client
        self._sessions = session_store
        self._cache = cache

    async def stream(
        self, user_id: str, message: str, image_b64: str | None = None, language: str = "es"
    ) -> AsyncGenerator[str, None]:
        blocked, pattern = is_injection(message)
        if blocked:
            logger.warning("Injection attempt from user %s: pattern=%r", user_id, pattern)
            yield safe_response(language)
            return

        session = await self._sessions.get(user_id)
        session.language = language

        if not image_b64:
            cached = await self._cache.get(message)
            if cached is not None:
                logger.debug("Semantic cache hit for user %s", user_id)
                yield cached
                return

        context = await self._mcp.build_context(user_id)

        # Augment context with semantically relevant memories for this message
        relevant = await self._mcp.search_memories(user_id, message, k=_SEARCH_K)
        if relevant:
            header = "### Relevant memories" if language == "en" else "### Recuerdos relevantes"
            snippets = [
                f"- [{r['layer']}] {r['content']}"
                for r in relevant
                if r.get("score", 0) >= _SEARCH_SCORE_MIN
            ]
            if snippets:
                context += f"\n\n{header}\n" + "\n".join(snippets)

        model_cfg = route_model(session.state, message, bool(image_b64))
        system_prompt = build_system_prompt(context, language)

        llm = make_llm(model_cfg.model, model_cfg.max_tokens)

        history_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
            for m in session.history
        ]

        current_human = HumanMessage(content=_build_human_content(message, image_b64))

        messages = [SystemMessage(content=system_prompt)] + history_messages + [current_human]

        full_response_chunks: list[str] = []

        async for chunk in llm.astream(messages):
            text = chunk.content
            if isinstance(text, list):
                # Vision responses can have list content blocks
                text = "".join(block.get("text", "") for block in text if isinstance(block, dict))
            if text:
                full_response_chunks.append(text)
                yield text

        full_response = "".join(full_response_chunks)

        # Post-response work runs in background so the stream closes immediately
        asyncio.create_task(
            self._post_response(user_id, message, full_response, session, image_b64)
        )

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

        # Memory upsert for mentioned events/context
        if _has_event_mention(message):
            await self._mcp.upsert_memory(
                user_id,
                "mentioned_events",
                {
                    "description": message[:120],
                    "message": message,
                    "response_preview": response[:200],
                },
            )
