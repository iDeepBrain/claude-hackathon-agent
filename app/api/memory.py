from fastapi import APIRouter, Request

router = APIRouter()

_LAYERS = ("mood_history", "mentioned_events", "habits", "interaction_prefs")


@router.get("/memory/{user_id}")
async def get_user_memory(user_id: str, request: Request):
    mcp_client = request.app.state.mcp_client
    raw = await mcp_client.get_memory(user_id)
    return {layer: raw.get(layer, [])[:5] for layer in _LAYERS}
