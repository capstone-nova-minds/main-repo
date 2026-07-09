"""Arabic text normalization helpers.

Used to clean raw OCR output before running extraction and NER, without
altering the meaning of names or values (over-normalization can break
name matching, so we keep changes conservative).
"""

import re

# Arabic-Indic and Extended Arabic-Indic digits -> ASCII digits
_ARABIC_DIGIT_MAP = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

# Alef variants -> plain alef
_ALEF_VARIANTS = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
})

# OCR garbage characters that are safe to strip (does not touch Arabic
# letters, digits, or common punctuation used in dates/case numbers).
_GARBAGE_PATTERN = re.compile(r"[|_~`\^\\<>{}\[\]]")

_MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")


def normalize_arabic_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII digits."""
    if not text:
        return text
    return "".join(_ARABIC_DIGIT_MAP.get(ch, ch) for ch in text)


def normalize_arabic_letters(text: str) -> str:
    """Normalize alef variants (أ, إ, آ -> ا) to help keyword/name matching."""
    if not text:
        return text
    return text.translate(_ALEF_VARIANTS)


def remove_garbage_symbols(text: str) -> str:
    """Strip common OCR noise symbols that never appear in legal text."""
    if not text:
        return text
    return _GARBAGE_PATTERN.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and trim each line."""
    if not text:
        return text
    lines = [_MULTI_SPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line != "")


def normalize_text(raw_text: str) -> dict:
    """Return both the original OCR text and a cleaned version.

    Cleaning order matters: digits/letters first, then garbage removal,
    then whitespace collapsing last so stripped symbols don't leave gaps.
    """
    if raw_text is None:
        raw_text = ""

    cleaned = normalize_arabic_digits(raw_text)
    cleaned = normalize_arabic_letters(cleaned)
    cleaned = remove_garbage_symbols(cleaned)
    cleaned = collapse_whitespace(cleaned)

    return {
        "original_text": raw_text,
        "cleaned_text": cleaned,
    }
