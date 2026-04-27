import logging

from fastapi import APIRouter, Request
from app.agent.llm import make_llm
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.persona import build_system_prompt

logger = logging.getLogger(__name__)

router = APIRouter()

_TRIGGER_PROMPTS: dict[str, dict[str, str]] = {
    "silence_anomaly": {
        "es": (
            "El usuario no ha enviado mensajes desde hace un tiempo inusual para él. "
            "Genera un mensaje corto y natural para retomar el contacto. "
            "No seas dramático ni alarmista. Algo simple y cálido."
        ),
        "en": (
            "The user hasn't sent any messages for an unusually long time for them. "
            "Write a short, natural message to reconnect. "
            "Don't be dramatic or alarming. Something simple and warm."
        ),
    },
    "morning_checkin": {
        "es": (
            "Es por la mañana. Genera un saludo breve y personalizado para el usuario "
            "basándote en el contexto que tienes sobre él. Máximo 2 frases."
        ),
        "en": (
            "It's morning. Write a brief, personalised greeting for the user "
            "based on the context you have about them. Two sentences at most."
        ),
    },
    "event_followup": {
        "es": (
            "El usuario mencionó un evento o situación importante que debía ocurrir pronto. "
            "Pregunta cómo fue de forma natural y específica, como si lo recordaras porque te importa."
        ),
        "en": (
            "The user mentioned an important upcoming event or situation. "
            "Ask how it went in a natural, specific way — as if you remember because you care."
        ),
    },
}

_DEFAULT_TRIGGER: dict[str, str] = {
    "es": "Genera un mensaje proactivo corto y cálido para el usuario.",
    "en": "Write a short, warm proactive message for the user.",
}


class TriggerRequest(BaseModel):
    user_id: str
    trigger_type: str  # morning_checkin | silence_anomaly | event_followup


@router.post("/trigger")
async def trigger(req: TriggerRequest, request: Request):
    mcp_client = request.app.state.mcp_client
    session_store = request.app.state.session_store

    async def generate():
        session = await session_store.get(req.user_id)
        language = session.language

        trigger_prompts = _TRIGGER_PROMPTS.get(req.trigger_type, {})
        trigger_instruction = trigger_prompts.get(language) or _DEFAULT_TRIGGER.get(language, _DEFAULT_TRIGGER["es"])

        context = await mcp_client.build_context(req.user_id)
        system_prompt = build_system_prompt(context, language)

        llm = make_llm("claude-haiku-4-5-20251001", 512)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=trigger_instruction),
        ]

        async for chunk in llm.astream(messages):
            text = chunk.content
            if isinstance(text, list):
                text = "".join(block.get("text", "") for block in text if isinstance(block, dict))
            if text:
                yield {"data": text}

    return EventSourceResponse(generate())
