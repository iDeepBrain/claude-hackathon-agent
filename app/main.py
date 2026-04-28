import asyncio
import logging
import os
from contextlib import asynccontextmanager

from app.agent.llm import discover_provider

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.chain import AlmaChain
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.config import router as config_router
from app.api.cron import router as cron_router
from app.api.demo import router as demo_router
from app.api.memory import router as memory_router
from app.api.proactivity import router as proactivity_router
from app.cache.semantic import SemanticCache
from app.cache.session import RedisSession
from app.mcp_client.client import MCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        llm_info = await discover_provider()
    except RuntimeError as exc:
        logger.error("LLM discovery failed: %s", exc)
        llm_info = {"provider": "none", "model": None, "preferred": "claude-opus-4-7", "fallback_reason": str(exc)}
    app.state.llm_info = llm_info
    logger.info("LLM elected at startup: %s / %s", llm_info.get("provider"), llm_info.get("model"))

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

    # Pre-create the redis client used by both APScheduler (local) and
    # Cloud Scheduler endpoints (production) so they share the same connection.
    import redis.asyncio as aioredis
    app.state.scheduler_redis = aioredis.from_url(redis_url)

    scheduler = None
    if os.getenv("SCHEDULER_ENABLED", "false").lower() == "true":
        from app.scheduler.proactive import create_scheduler
        scheduler = create_scheduler(app.state.scheduler_redis)
        scheduler.start()
        logger.info("Proactive scheduler (in-process APScheduler) started")
    else:
        logger.info("Proactive scheduler disabled — Cloud Scheduler will hit /cron/proactive/{slot}")

    logger.info("Alma Agent started")
    yield
    if scheduler:
        scheduler.shutdown(wait=False)
    logger.info("Alma Agent shutting down")


app = FastAPI(title="Alma Agent", lifespan=lifespan)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(memory_router, prefix="/api/v1")
app.include_router(proactivity_router, prefix="/api/v1")
app.include_router(demo_router, prefix="/api/v1")  # POST /api/v1/demo/seed
app.include_router(auth_router, prefix="/api/v1")  # POST /api/v1/auth/google
app.include_router(config_router, prefix="/api/v1")  # GET /api/v1/config (public client config)
app.include_router(cron_router)  # Cloud Scheduler: /cron/proactive/{slot}


@app.get("/health")
async def health(request: Request):
    info = getattr(request.app.state, "llm_info", {})
    return {
        "status": "ok",
        "provider": info.get("provider", "unknown"),
        "model": info.get("model"),
        "preferred": info.get("preferred", "claude-opus-4-7"),
        "fallback_reason": info.get("fallback_reason"),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})
