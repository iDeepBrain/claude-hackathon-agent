from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_PERSONA_FILES: dict[str, str] = {
    "es": "alma_es.md",
    "en": "alma_en.md",
}
_DEFAULT_LANGUAGE = "es"


def load_persona(language: str = _DEFAULT_LANGUAGE) -> str:
    filename = _PERSONA_FILES.get(language, _PERSONA_FILES[_DEFAULT_LANGUAGE])
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def build_system_prompt(context: str, language: str = _DEFAULT_LANGUAGE) -> str:
    persona = load_persona(language)
    if context and context != "Sin historial previo para este usuario.":
        section_header = "## Current user context" if language == "en" else "## Contexto actual del usuario"
        return f"{persona}\n\n{section_header}\n{context}"
    return persona
