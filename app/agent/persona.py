import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_PERSONA_FILES: dict[str, str] = {
    "es": "alma_es.md",
    "en": "alma_en.md",
}
_DEFAULT_LANGUAGE = "es"

# Each prompt file starts with an HTML metadata block holding the version,
# framework tags, and citation discipline. That block is for humans
# (review, fixture replay, plugin tooling) — not the LLM. Strip it before
# we pass the persona into the system prompt so we don't waste tokens or
# accidentally leak versioning detail into model behavior.
_METADATA_HEADER_RE = re.compile(r"^\s*<!--.*?-->\s*", re.DOTALL)


def load_persona(language: str = _DEFAULT_LANGUAGE) -> str:
    filename = _PERSONA_FILES.get(language, _PERSONA_FILES[_DEFAULT_LANGUAGE])
    raw = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    return _METADATA_HEADER_RE.sub("", raw, count=1)


def build_system_prompt(context: str, language: str = _DEFAULT_LANGUAGE) -> str:
    persona = load_persona(language)
    if context and context != "Sin historial previo para este usuario.":
        section_header = "## Current user context" if language == "en" else "## Contexto actual del usuario"
        return f"{persona}\n\n{section_header}\n{context}"
    return persona
