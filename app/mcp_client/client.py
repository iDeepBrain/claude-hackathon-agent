import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT = "Sin historial previo para este usuario."


class MCPClient:
    def __init__(self, mcp_url: str) -> None:
        self._url = mcp_url
        self._lc_client = MultiServerMCPClient(
            {
                "memory": {
                    "url": mcp_url,
                    "transport": "streamable_http",
                }
            }
        )
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}

    async def get_tools(self) -> list[BaseTool]:
        """Load LangChain tools from the MCP server.

        Called once at startup; result cached in _tool_map for direct invocation.
        get_tools() opens a new session per call, which is fine for initialization.
        """
        try:
            self._tools = await self._lc_client.get_tools()
            self._tool_map = {t.name: t for t in self._tools}
            logger.info("Loaded %d MCP tools: %s", len(self._tools), list(self._tool_map.keys()))
        except Exception as exc:
            logger.warning("MCP server unreachable at startup: %s", exc)
        return self._tools

    async def _invoke(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Invoke a named tool, returning None on any failure."""
        tool = self._tool_map.get(tool_name)
        if tool is None:
            logger.warning("Tool %r not found in MCP tool map", tool_name)
            return None
        try:
            return await tool.ainvoke(args)
        except Exception as exc:
            logger.warning("MCP tool %r failed: %s", tool_name, exc)
            return None

    async def build_context(self, user_id: str) -> str:
        result = await self._invoke("build_context_tool", {"user_id": user_id})
        if result is None:
            return _DEFAULT_CONTEXT
        return result if isinstance(result, str) else str(result)

    async def search_memories(self, user_id: str, query: str, k: int = 5) -> list:
        result = await self._invoke("search_memories_tool", {"user_id": user_id, "query": query, "k": k})
        if result is None:
            return []
        return result if isinstance(result, list) else []

    async def upsert_memory(self, user_id: str, layer: str, data: dict) -> dict:
        result = await self._invoke("upsert_memory_tool", {"user_id": user_id, "layer": layer, "data": data})
        if result is None:
            return {"success": False}
        return result if isinstance(result, dict) else {"success": False}

    async def evaluate_crisis_risk(self, user_id: str, message: str) -> dict:
        result = await self._invoke("evaluate_crisis_risk_tool", {"user_id": user_id, "message": message})
        if result is None:
            return {"score": 0.0, "level": "none"}
        return result if isinstance(result, dict) else {"score": 0.0, "level": "none"}

    async def get_memory(self, user_id: str) -> dict:
        result = await self._invoke("get_memory_tool", {"user_id": user_id})
        if result is None:
            return {"mood_history": [], "mentioned_events": [], "habits": [], "interaction_prefs": []}
        return result if isinstance(result, dict) else {"mood_history": [], "mentioned_events": [], "habits": [], "interaction_prefs": []}
