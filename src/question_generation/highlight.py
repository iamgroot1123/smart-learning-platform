import random
import re
from typing import Optional

# Try to import spaCy lazily — if not installed we'll fall back later
try:
    import spacy
    from spacy.lang.en import English  # for type hints only
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load spaCy model and cache it on the module.
    If the model package is not installed, this function will raise ImportError.
    """
    global _SPACY_AVAILABLE, _NLP
    if not _SPACY_AVAILABLE:
        raise ImportError("spaCy not installed")

    try:
        # prefer to load a previously installed package
        _NLP = spacy.load(model_name)
    except OSError as e:
        # model not found — raise with helpful message
        raise OSError(
            f"spaCy model '{model_name}' not found. Install it with:\n"
            f"  python -m spacy download {model_name}\n"
            "or add it to requirements and install before running the app."
        ) from e

    return _NLP


# lazy holder for the loaded model
_NLP = None


def _ensure_nlp(model_name: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        _NLP = load_spacy_model(model_name)
    return _NLP


def highlight_answer_spacy(text: str, model_name: str = "en_core_web_sm") -> str:
    """
    Use spaCy to pick a good answer candidate and highlight it:
    wraps selected text with <answer>...</answer>
    """
    if not _SPACY_AVAILABLE:
        raise ImportError("spaCy is not available")

    nlp = _ensure_nlp(model_name)
    doc = nlp(text)

    # 1) Try noun chunks (multi-word nouns) — prefer longer chunk
    noun_chunks = [nc.text.strip() for nc in doc.noun_chunks if len(nc.text.strip()) > 2]
    if noun_chunks:
        # prioritize by length (more informative) then randomness
        noun_chunks.sort(key=lambda s: (-len(s.split()), -len(s)))
        choice = noun_chunks[0]
        return _wrap_first_occurrence(text, choice)

    # 2) Try proper nouns / named entities (PERSON, ORG, GPE, LOC, PRODUCT...)
    ents = [ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}]
    if ents:
        # pick longest
        ents.sort(key=lambda s: (-len(s.split()), -len(s)))
        return _wrap_first_occurrence(text, ents[0])

    # 3) Try noun tokens (NN, NNS, NNPS, NNP)
    nouns = [tok.text for tok in doc if tok.pos_ == "NOUN" or tok.pos_ == "PROPN"]
    if nouns:
        # pick the longest noun token
        nouns = [n for n in nouns if len(n) > 3]
        if nouns:
            nouns.sort(key=lambda s: (-len(s), s))
            return _wrap_first_occurrence(text, nouns[0])

    # 4) fallback to a regex-based pick (long-ish words)
    return _wrap_first_occurrence_regex(text)


def _wrap_first_occurrence(text: str, substr: str) -> str:
    """Replace only the first exact word-match occurrence of substr with the highlighted tag."""
    # escape for regex and use word boundaries
    esc = re.escape(substr)
    pattern = rf"\b{esc}\b"
    # if that exact pattern not found, fallback to a simple substring replacement
    if re.search(pattern, text):
        return re.sub(pattern, f"<answer>{substr}</answer>", text, count=1)
    else:
        # fallback: first substring match (less strict)
        return text.replace(substr, f"<answer>{substr}</answer>", 1)


def _wrap_first_occurrence_regex(text: str) -> str:
    """Regex fallback: pick a decent long word and highlight it."""
    words = re.findall(r"\w{5,}", text)  # words with length >=5
    if not words:
        return text  # nothing to highlight
    # prefer longer words
    words.sort(key=lambda w: (-len(w), w))
    chosen = words[0]
    return _wrap_first_occurrence(text, chosen)


# Convenience wrapper used by the app (with safe fallback)
def highlight_answer(text: str, use_spacy: bool = True, model_name: str = "en_core_web_sm") -> str:
    """
    Try spaCy if requested and available, else use regex fallback.
    Returns the text with a selected answer wrapped in <answer>...</answer>.
    """
    if use_spacy and _SPACY_AVAILABLE:
        try:
            return highlight_answer_spacy(text, model_name=model_name)
        except Exception:
            # any spaCy failure -> fallback safely
            return _wrap_first_occurrence_regex(text)
    else:
        return _wrap_first_occurrence_regex(text)
