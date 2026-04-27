from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: str | None = None
    language: str = "es"  # es | en


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    alma_chain = request.app.state.alma_chain

    async def generate():
        async for chunk in alma_chain.stream(req.user_id, req.message, req.image_base64, req.language):
            yield {"data": chunk}

    return EventSourceResponse(generate())
