#!/usr/bin/env python3
"""reset_demo.py — Local CLI to seed the demo user via the MCP server.

Usage (from claude-hackathon-agent root):

    python scripts/reset_demo.py
    python scripts/reset_demo.py --user-id demo_mateo --mcp-url http://localhost:8001/mcp

Run this whenever you want a fresh, populated memory state for the demo user.
The script is idempotent — running twice gives the same final state.

This is the developer-facing entry point; the production-facing entry point
is the POST /api/v1/demo/seed endpoint, which is protected by DEMO_SEED_TOKEN.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure `app` package is importable when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_ROOT = os.path.dirname(_THIS_DIR)
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from app.mcp_client.client import MCPClient
from app.seed_demo import DEMO_USER_ID_DEFAULT, seed_mateo


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", default=DEMO_USER_ID_DEFAULT, help="User ID to seed (default: %(default)s)")
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("MCP_URL", "http://localhost:8001/mcp"),
        help="MCP server URL (default: $MCP_URL or http://localhost:8001/mcp)",
    )
    args = parser.parse_args()

    print(f"Seeding demo user {args.user_id!r} via MCP at {args.mcp_url}")
    mcp = MCPClient(args.mcp_url)
    tools = await mcp.get_tools()
    if not tools:
        print(f"ERROR: MCP server at {args.mcp_url} did not return tools — is it running?", file=sys.stderr)
        return 2

    counts = await seed_mateo(mcp, args.user_id)

    print("\nSeed complete:")
    for layer, count in counts.items():
        print(f"  {layer:20s} {count} entries")
    print(f"\nVerify: curl http://localhost:8080/api/v1/memory/{args.user_id}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
