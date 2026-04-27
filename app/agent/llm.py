import logging
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_provider: str = "anthropic"

_GEMINI_EQUIV: dict[str, str] = {
    "claude-haiku-4-5-20251001": "gemini-2.0-flash",
    "claude-sonnet-4-6": "gemini-1.5-pro",
    "claude-opus-4-7": "gemini-1.5-pro",
}


def set_provider(provider: str) -> None:
    global _provider
    _provider = provider
    logger.info("LLM provider → %s", provider)


def get_provider() -> str:
    return _provider


def make_llm(model: str, max_tokens: int) -> BaseChatModel:
    if _provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_GEMINI_EQUIV.get(model, "gemini-2.0-flash"),
            max_output_tokens=max_tokens,
        )
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, max_tokens=max_tokens, streaming=True)


def build_image_content(text: str, image_b64: str) -> list:
    if _provider == "gemini":
        return [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
        {"type": "text", "text": text},
    ]
