"""Demo seed endpoint — populates the canonical demo user with realistic memory.

Used to make cold demo sessions show populated state instead of empty memory
panels. Protected by the DEMO_SEED_TOKEN env var (sent as X-Demo-Token header).
If the env var is unset, the endpoint is disabled and returns 503.

Pattern mirrors /cron/proactive/{slot} — header-based bearer auth, idempotent
operation, structured response.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request

from app.seed_demo import DEMO_USER_ID_DEFAULT, seed_mateo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["demo"])

EXPECTED_TOKEN = os.getenv("DEMO_SEED_TOKEN", "")


def _verify_token(token: str | None) -> None:
    if not EXPECTED_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Demo seed endpoint disabled (DEMO_SEED_TOKEN env not configured)",
        )
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid X-Demo-Token")


@router.post("/demo/seed")
async def demo_seed(
    request: Request,
    user_id: str = DEMO_USER_ID_DEFAULT,
    x_demo_token: str | None = Header(default=None, alias="X-Demo-Token"),
):
    """Re-seed the demo user with the canonical Mateo persona.

    Returns the count of memory records written per layer. Idempotent.
    """
    _verify_token(x_demo_token)

    mcp_client = request.app.state.mcp_client
    counts = await seed_mateo(mcp_client, user_id)

    logger.info("demo/seed executed for user_id=%r", user_id)
    return {
        "status": "seeded",
        "user_id": user_id,
        "layers": counts,
    }
