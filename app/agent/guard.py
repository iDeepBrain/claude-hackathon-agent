import unicodedata

from rapidfuzz import fuzz, process


def _normalize(text: str) -> str:
    """Lowercase + NFC normalization so accented chars match regardless of encoding."""
    return unicodedata.normalize("NFC", text).lower()


INJECTION_PATTERNS = [
    # ── English — long (fuzzy-eligible) ───────────────────────────────────────
    "ignore previous instructions",
    "forget your instructions",
    "pretend you are",
    "override instructions",
    "disregard all instructions",
    "simulate being an ai",
    "from now on you are",
    "sin restricciones de contenido",
    # ── English — short (exact-match only) ───────────────────────────────────
    "act as",
    "new role",
    "you are now",
    "now you are",
    "disregard",
    "jailbreak",
    "system prompt",
    "system:",
    "ignore all",
    "roleplay as",
    "role play as",
    "bypass",
    "override all",
    "forget all",
    "as if you were",
    "simulate being",
    # ── Spanish — long (fuzzy-eligible) ──────────────────────────────────────
    "instrucciones anteriores",
    "olvida tus instrucciones",
    "ignora todo lo anterior",
    "ignora tus instrucciones",
    # ── Spanish — short (exact-match only) ───────────────────────────────────
    "actúa como",
    "actua como",           # without accent (common typo)
    "nuevo rol",
    "eres ahora",
    "ahora eres",
    "sin censura",
    "sin restricciones",
    "sin reglas",
    "modo libre",
    "modo sin restricciones",
]

# Pre-normalize all patterns once at import time
_NORMALIZED_PATTERNS = [(_normalize(p), p) for p in INJECTION_PATTERNS]

_SAFE_RESPONSES: dict[str, str] = {
    "es": "Lo siento, no pude procesar ese mensaje. ¿Puedes intentarlo de otra forma?",
    "en": "Sorry, I wasn't able to process that message. Can you try rephrasing it?",
}


def safe_response(language: str = "es") -> str:
    return _SAFE_RESPONSES.get(language, _SAFE_RESPONSES["es"])


_FUZZY_THRESHOLD = 88
_FUZZY_MIN_LEN = 16  # patterns shorter than this are only exact-matched


def is_injection(message: str) -> tuple[bool, str]:
    """Returns (is_blocked, matched_pattern).

    Pass 1: exact substring against every normalized pattern.
    Pass 2: partial_ratio only on patterns >= _FUZZY_MIN_LEN — catches typos/OCR
            errors while avoiding false positives from WRatio's token-based scoring.
    """
    normalized = _normalize(message)

    for norm_pat, original_pat in _NORMALIZED_PATTERNS:
        if norm_pat in normalized:
            return True, original_pat

    fuzzy_candidates = [norm_pat for norm_pat, _ in _NORMALIZED_PATTERNS if len(norm_pat) >= _FUZZY_MIN_LEN]
    result = process.extractOne(
        normalized,
        fuzzy_candidates,
        scorer=fuzz.partial_ratio,
        score_cutoff=_FUZZY_THRESHOLD,
    )
    if result is not None:
        matched_norm, _score, _idx = result
        original = next(op for np_, op in _NORMALIZED_PATTERNS if np_ == matched_norm)
        return True, original

    return False, ""
