import json
import redis.asyncio as aioredis
from dataclasses import dataclass, field

SESSION_KEY = "alma:session:{user_id}"


@dataclass
class Session:
    state: str = "onboarding"  # onboarding | chat | crisis
    language: str = "es"       # es | en
    history: list = field(default_factory=list)  # [{role: human|assistant, content: str}]


class RedisSession:
    def __init__(self, redis_url: str, ttl: int = 86400) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl

    async def get(self, user_id: str) -> Session:
        key = SESSION_KEY.format(user_id=user_id)
        raw = await self._redis.get(key)
        if raw is None:
            return Session()
        data = json.loads(raw)
        return Session(
            state=data.get("state", "onboarding"),
            language=data.get("language", "es"),
            history=data.get("history", []),
        )

    async def save(self, user_id: str, session: Session) -> None:
        key = SESSION_KEY.format(user_id=user_id)
        payload = json.dumps({"state": session.state, "language": session.language, "history": session.history})
        await self._redis.set(key, payload, ex=self._ttl)

    async def update_state(self, user_id: str, state: str) -> None:
        session = await self.get(user_id)
        session.state = state
        await self.save(user_id, session)

    def _trim_history(self, history: list, max_turns: int = 20) -> list:
        return history[-max_turns:]
