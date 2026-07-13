"""Arabic text normalization helpers.

Used to clean raw OCR output before running extraction and NER, without
altering the meaning of names or values.
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

# OCR garbage characters that are safe to strip.
# Important: do NOT remove "/" because it is needed for dates and case numbers.
_GARBAGE_PATTERN = re.compile(r"[|_~`\^<>{}\[\]]")

# EasyOCR consistently misreads the "W" in this document system's "UW-YYYY-NNNN"
# document/book number as a vertical bar, e.g. "U|-2026-0101" instead of
# "UW-2026-0101" -- seen in every sample of this document type. Restore it
# *before* the generic garbage-symbol strip below deletes the "|" and the
# "W" is lost for good. Scoped tightly (single Latin letter + "|" + "-" +
# digit) so it can never touch the "|" used as a field separator elsewhere
# (e.g. "98000000081 | نوع الشخص"), which never has a letter immediately
# before it or a hyphen-digit immediately after.
_MISREAD_W_PATTERN = re.compile(r"(?<=[A-Za-z])\|(?=-\d)")

_MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")


def normalize_arabic_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII digits."""
    if not text:
        return text
    return "".join(_ARABIC_DIGIT_MAP.get(ch, ch) for ch in text)


def normalize_arabic_letters(text: str) -> str:
    """Normalize alef variants: أ, إ, آ -> ا."""
    if not text:
        return text
    return text.translate(_ALEF_VARIANTS)


def normalize_separators(text: str) -> str:
    """Normalize separators used in dates and case numbers."""
    if not text:
        return text

    replacements = {
        "\\": "/",   # OCR may read slash as backslash
        "／": "/",
        "⁄": "/",
        "–": "-",
        "—": "-",
        "−": "-",
        "：": ":",
        "؛": ":",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_garbage_symbols(text: str) -> str:
    """Strip common OCR noise symbols that never appear in legal text."""
    if not text:
        return text
    text = _MISREAD_W_PATTERN.sub("W", text)
    return _GARBAGE_PATTERN.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and trim each line."""
    if not text:
        return text

    lines = [
        _MULTI_SPACE_PATTERN.sub(" ", line).strip()
        for line in text.splitlines()
    ]

    return "\n".join(line for line in lines if line != "")


def normalize_arabic_text(text: str) -> str:
    """
    Return cleaned Arabic text as a plain string.
    This is the main function used by extraction services.
    """
    if text is None:
        text = ""

    cleaned = normalize_arabic_digits(text)
    cleaned = normalize_arabic_letters(cleaned)
    cleaned = normalize_separators(cleaned)
    cleaned = remove_garbage_symbols(cleaned)
    cleaned = collapse_whitespace(cleaned)

    return cleaned.strip()


def split_lines(text: str) -> list[str]:
    """Normalize text and split it into non-empty lines."""
    cleaned = normalize_arabic_text(text)
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def normalize_text(raw_text: str) -> dict:
    """
    Return both original OCR text and cleaned text.
    Keep this function because process.py may already depend on it.
    """
    if raw_text is None:
        raw_text = ""

    cleaned = normalize_arabic_text(raw_text)

    return {
        "original_text": raw_text,
        "cleaned_text": cleaned,
    }