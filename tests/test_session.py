import pytest
import fakeredis.aioredis as fakeredis
from app.cache.session import RedisSession, Session


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def session_store(fake_redis):
    store = RedisSession.__new__(RedisSession)
    store._redis = fake_redis
    store._ttl = 86400
    return store


async def test_get_session_new_user(session_store):
    session = await session_store.get("user_new")
    assert session.state == "onboarding"
    assert session.history == []
    assert session.language == "es"


async def test_save_and_get_session(session_store):
    s = Session(state="chat", history=[{"role": "human", "content": "hola"}])
    await session_store.save("user1", s)
    loaded = await session_store.get("user1")
    assert loaded.state == "chat"
    assert len(loaded.history) == 1
    assert loaded.history[0]["content"] == "hola"


async def test_update_state(session_store):
    await session_store.save("user2", Session(state="chat"))
    await session_store.update_state("user2", "crisis")
    loaded = await session_store.get("user2")
    assert loaded.state == "crisis"


def test_trim_history(session_store):
    history = [{"role": "human", "content": f"msg {i}"} for i in range(50)]
    trimmed = session_store._trim_history(history, max_turns=20)
    assert len(trimmed) == 20
    assert trimmed[-1]["content"] == "msg 49"


async def test_session_persists_across_calls(session_store):
    s = Session(state="chat", history=[{"role": "human", "content": "primero"}])
    await session_store.save("user3", s)
    s2 = await session_store.get("user3")
    s2.history.append({"role": "assistant", "content": "respuesta"})
    await session_store.save("user3", s2)
    s3 = await session_store.get("user3")
    assert len(s3.history) == 2


# ── language field ─────────────────────────────────────────────────────────────

async def test_language_persists(session_store):
    s = Session(state="chat", language="en")
    await session_store.save("user_en", s)
    loaded = await session_store.get("user_en")
    assert loaded.language == "en"


async def test_language_default_es(session_store):
    s = Session(state="chat")
    await session_store.save("user_default", s)
    loaded = await session_store.get("user_default")
    assert loaded.language == "es"


async def test_language_update(session_store):
    s = Session(state="chat", language="es")
    await session_store.save("user_lang", s)

    s2 = await session_store.get("user_lang")
    s2.language = "en"
    await session_store.save("user_lang", s2)

    s3 = await session_store.get("user_lang")
    assert s3.language == "en"


async def test_missing_language_field_defaults_to_es(session_store):
    """Sessions saved before language field existed should not crash."""
    import json
    key = "alma:session:legacy_user"
    payload = json.dumps({"state": "chat", "history": []})  # no language key
    await session_store._redis.set(key, payload)
    loaded = await session_store.get("legacy_user")
    assert loaded.language == "es"
