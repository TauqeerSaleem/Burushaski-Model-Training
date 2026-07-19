import re
import string
import unicodedata


_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",
}


def normalize_burushaski_text(text: str | None) -> str:
    """A light normalization pass. Real dialect differences should stay intact."""
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", str(text))
    for old, new in _APOSTROPHES.items():
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_english_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = str(text).lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    return " ".join(cleaned.split())
