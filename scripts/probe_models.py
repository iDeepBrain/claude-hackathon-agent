"""Probe Gemini models to find which ones still work for our use case.

Run inside the agent container:
    docker exec claude-hackathon-infra-agent-1 python scripts/probe_models.py

Or directly with .venv:
    source .venv/bin/activate && python scripts/probe_models.py
"""
import asyncio
import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI

# Candidates ordered cheapest/fastest first.
CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash-thinking-exp",
    "gemini-exp-1206",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",  # known deprecated, here as control
]


async def probe(model: str) -> tuple[str, bool, str]:
    try:
        llm = ChatGoogleGenerativeAI(model=model, max_output_tokens=20)
        t0 = time.monotonic()
        response = await asyncio.wait_for(
            llm.ainvoke("Say 'hello' in Spanish in 3 words."),
            timeout=15.0,
        )
        elapsed = time.monotonic() - t0
        text = (response.content or "").strip()[:60]
        return model, True, f"{elapsed:.2f}s | {text}"
    except asyncio.TimeoutError:
        return model, False, "TIMEOUT (>15s)"
    except Exception as exc:
        return model, False, f"{type(exc).__name__}: {str(exc)[:120]}"


async def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("✗ GOOGLE_API_KEY missing in env. Aborting.")
        return

    print(f"Probing {len(CANDIDATES)} Gemini models...\n")
    results = await asyncio.gather(*(probe(m) for m in CANDIDATES))

    print(f"{'Model':<42} {'Status':<8} Detail")
    print("─" * 100)
    for model, ok, detail in results:
        mark = "✓ PASS" if ok else "✗ FAIL"
        print(f"{model:<42} {mark:<8} {detail}")

    working = [m for m, ok, _ in results if ok]
    print(f"\nWorking: {len(working)}/{len(CANDIDATES)}")
    if working:
        print("Recommended for sonnet/opus replacement:", working[0])


if __name__ == "__main__":
    asyncio.run(main())
