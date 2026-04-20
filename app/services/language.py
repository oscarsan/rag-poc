from __future__ import annotations

from app.domain import Language

# Characters that strongly indicate Finnish (diacritics common in Finnish).
_FI_CHARS = set("äöÄÖ")

# Very common short Finnish function words that rarely appear in English text.
_FI_WORDS = {
    "ja", "on", "ei", "että", "mitä", "mikä", "miten", "missä", "milloin",
    "kuinka", "minä", "sinä", "hän", "me", "te", "he", "olen", "oletko",
    "voiko", "voi", "kiitos", "paljonko", "hinta", "hintoja", "aukioloajat",
    "aamiainen", "huone", "varaus",
}


def detect_language(text: str) -> Language:
    """Tiny heuristic detector: returns 'fi' or 'en'.

    Rationale: for a PoC the inputs are short guest questions. A proper
    detector (langdetect, fasttext) is overkill; accuracy here is good enough
    because Finnish almost always contains ä/ö or telltale function words.
    """
    if not text:
        return "en"
    if any(ch in _FI_CHARS for ch in text):
        return "fi"
    tokens = {t.strip(".,!?¿¡;:\"'()[]").lower() for t in text.split()}
    if tokens & _FI_WORDS:
        return "fi"
    return "en"
