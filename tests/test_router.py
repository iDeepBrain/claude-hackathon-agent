from app.agent.router import ModelConfig, route_model


def test_default_routes_to_haiku():
    cfg = route_model("chat", "hola")
    assert cfg.model == "claude-haiku-4-5-20251001"


def test_crisis_state_routes_to_opus():
    cfg = route_model("crisis", "estoy bien")
    assert cfg.model == "claude-opus-4-7"


def test_image_routes_to_haiku():
    cfg = route_model("chat", "mira esta foto", has_image=True)
    assert cfg.model == "claude-haiku-4-5-20251001"


def test_long_message_routes_to_sonnet():
    long_msg = "x" * 801
    cfg = route_model("chat", long_msg)
    assert cfg.model == "claude-sonnet-4-6"


def test_crisis_overrides_image():
    cfg = route_model("crisis", "imagen", has_image=True)
    assert cfg.model == "claude-opus-4-7"


def test_model_config_is_dataclass():
    cfg = route_model("onboarding", "hola")
    assert isinstance(cfg, ModelConfig)
    assert isinstance(cfg.max_tokens, int)
    assert cfg.max_tokens > 0
