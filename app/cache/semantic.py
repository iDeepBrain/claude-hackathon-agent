import hashlib
import json
import logging

import numpy as np
import redis.asyncio as aioredis
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

_RECENT_MSGS_KEY = "alma:cache:recent_msgs"
_EMB_KEY_PREFIX = "alma:cache:emb:"
_RECENT_MAX = 100
_FUZZY_THRESHOLD = 88


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _msg_hash(message: str) -> str:
    return hashlib.sha256(message.encode()).hexdigest()[:16]


class SemanticCache:
    def __init__(self, redis_url: str, ttl: int = 3600, threshold: float = 0.92) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl
        self._threshold = threshold
        self._model = None

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vec = next(self._model.embed([text])).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    async def get(self, message: str) -> str | None:
        # Stage 1: rapidfuzz pre-filter against recent message list
        raw_recent = await self._redis.lrange(_RECENT_MSGS_KEY, 0, _RECENT_MAX - 1)
        recent_msgs = [r.decode() for r in raw_recent]

        candidate_key: str | None = None
        for candidate in recent_msgs:
            if fuzz.WRatio(message, candidate) > _FUZZY_THRESHOLD:
                candidate_key = _msg_hash(candidate)
                break

        if candidate_key is None:
            return None

        # Stage 2: embedding cosine similarity
        emb_key = f"{_EMB_KEY_PREFIX}{candidate_key}"
        stored_raw = await self._redis.get(emb_key)
        if stored_raw is None:
            return None

        stored = json.loads(stored_raw.decode())
        cached_emb = np.array(stored["embedding"], dtype=np.float32)
        query_emb = self._embed(message)

        if _cosine_similarity(query_emb, cached_emb) >= self._threshold:
            logger.debug("Semantic cache hit for message hash %s", candidate_key)
            return stored["response"]

        return None

    async def set(self, message: str, response: str) -> None:
        emb = self._embed(message)
        key_hash = _msg_hash(message)
        emb_key = f"{_EMB_KEY_PREFIX}{key_hash}"

        payload = json.dumps({"response": response, "embedding": emb.tolist()})
        await self._redis.set(emb_key, payload.encode(), ex=self._ttl)

        # Push to recent list and cap length; decode=False so we push raw bytes
        await self._redis.lpush(_RECENT_MSGS_KEY, message.encode())
        await self._redis.ltrim(_RECENT_MSGS_KEY, 0, _RECENT_MAX - 1)
        await self._redis.expire(_RECENT_MSGS_KEY, self._ttl)
