from dataclasses import dataclass

_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"
_OPUS = "claude-opus-4-7"


@dataclass
class ModelConfig:
    model: str
    max_tokens: int = 1024


def route_model(session_state: str, message: str, has_image: bool = False) -> ModelConfig:
    if session_state == "crisis":
        return ModelConfig(model=_OPUS, max_tokens=2048)
    if has_image:
        return ModelConfig(model=_HAIKU, max_tokens=1024)
    if len(message) > 800:
        return ModelConfig(model=_SONNET, max_tokens=2048)
    return ModelConfig(model=_HAIKU, max_tokens=1024)
