import asyncio
import logging
import os
from contextlib import asynccontextmanager

from app.agent.llm import set_provider

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.chain import AlmaChain
from app.api.chat import router as chat_router
from app.api.memory import router as memory_router
from app.api.proactivity import router as proactivity_router
from app.cache.semantic import SemanticCache
from app.cache.session import RedisSession
from app.mcp_client.client import MCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


async def _check_anthropic() -> bool:
    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True
    except Exception as exc:
        logger.warning("Anthropic unavailable (%s: %s) — switching to Gemini fallback", type(exc).__name__, exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not await _check_anthropic():
        set_provider("gemini")
    else:
        logger.info("Anthropic API reachable — using Claude")

    redis_url = os.environ["REDIS_URL"]
    mcp_url = os.environ["MCP_URL"]
    session_ttl = int(os.getenv("SESSION_TTL", "86400"))
    cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    cache_threshold = float(os.getenv("CACHE_THRESHOLD", "0.92"))

    mcp_client = MCPClient(mcp_url)
    for attempt in range(6):
        tools = await mcp_client.get_tools()
        if tools:
            break
        wait = 2 ** attempt
        logger.warning("MCP not ready (attempt %d/6), retrying in %ds…", attempt + 1, wait)
        await asyncio.sleep(wait)

    session_store = RedisSession(redis_url, ttl=session_ttl)
    cache = SemanticCache(redis_url, ttl=cache_ttl, threshold=cache_threshold)
    alma_chain = AlmaChain(mcp_client, session_store, cache)

    app.state.mcp_client = mcp_client
    app.state.session_store = session_store
    app.state.cache = cache
    app.state.alma_chain = alma_chain

    scheduler = None
    if os.getenv("SCHEDULER_ENABLED", "false").lower() == "true":
        import redis.asyncio as aioredis
        from app.scheduler.proactive import create_scheduler
        scheduler_redis = aioredis.from_url(redis_url)
        scheduler = create_scheduler(scheduler_redis)
        scheduler.start()
        logger.info("Proactive scheduler started")

    logger.info("Alma Agent started")
    yield
    if scheduler:
        scheduler.shutdown(wait=False)
    logger.info("Alma Agent shutting down")


app = FastAPI(title="Alma Agent", lifespan=lifespan)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(memory_router, prefix="/api/v1")
app.include_router(proactivity_router, prefix="/api/v1")


@app.get("/health")
async def health():
    from app.agent.llm import get_provider
    return {"status": "ok", "provider": get_provider()}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})
