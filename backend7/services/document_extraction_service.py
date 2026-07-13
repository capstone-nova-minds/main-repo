"""Rule-based extraction of document-level fields.

Extracts:
- court_name
- case_number
- document_number
- document_date

No LLM, no cloud AI. Everything is deterministic and explainable.
"""

import re
import difflib
from typing import List, Optional

from schemas.document_schema import DocumentInfo, ExtractedField
from utils.confidence import (
    EXACT_MATCH_CONFIDENCE,
    NEARBY_MATCH_CONFIDENCE,
    MISSING_CONFIDENCE,
    needs_review,
)
from utils.known_courts import KNOWN_COURT_NAMES
from utils.regex_patterns import (
    COURT_NAME_KEYWORDS,
    CASE_NUMBER_KEYWORDS,
    DOCUMENT_NUMBER_KEYWORDS,
    DOCUMENT_DATE_KEYWORDS,
    DATE_PATTERN_YMD,
    DATE_PATTERN_DMY,
    DATE_PATTERN_COMPACT,
)


# ---------------------------------------------------------------------------
# Strong case / document number patterns
# ---------------------------------------------------------------------------

# Example:
# رقم القضية : 1204/2026
# 1204/2026 : رقم القضية
CASE_NUMBER_PATTERNS = [
    re.compile(
        r"(?:رقم\s*القضية|رقم\s*الدعوى|القضية\s*رقم|الدعوى\s*رقم|دعوى\s*رقم|بالدعوى\s*رقم|بدعوى\s*رقم|رقم\s*الفضية|الفضية\s*رقم)"
        r"\s*[:：.]?\s*"
        r"(?:\d\s+)?"  # skip a single stray junk digit (OCR misread colon)
        r"(?P<value>\d{1,6}\s*[/\s]\s*(?:20\d{2}|19\d{2})|(?:20\d{2}|19\d{2})\s*[/\s]\s*\d{1,6})"
    ),
    re.compile(
        r"(?P<value>\d{1,6}\s*[/\s]\s*(?:20\d{2}|19\d{2})|(?:20\d{2}|19\d{2})\s*[/\s]\s*\d{1,6})"
        r"\s*[:：.]?\s*"
        r"(?:رقم\s*القضية|رقم\s*الدعوى|القضية\s*رقم|الدعوى\s*رقم|دعوى\s*رقم|بالدعوى\s*رقم|بدعوى\s*رقم|رقم\s*الفضية|الفضية\s*رقم)"
    ),
]

# Example:
# رقم الكتاب : UW-2026-0004
# UW-2026-0004 : رقم الكتاب
DOCUMENT_NUMBER_PATTERNS = [
    re.compile(
        r"(?:رقم\s*الكتاب|كتاب\s*رقم|رقم\s*الصادر|الكتاب)"
        r"\s*[:：.]?\s*"
        r"(?:\d\s+)?"  # skip a single stray junk digit (OCR misread colon)
        r"(?P<value>[A-Za-z0-9|/\s]{1,6}[-–—]\s*\d{4}\s*[-–—]\s*\d{3,10})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>[A-Za-z0-9|/\s]{1,6}[-–—]\s*\d{4}\s*[-–—]\s*\d{3,10})"
        r"\s*[:：.]?\s*"
        r"(?:رقم\s*الكتاب|كتاب\s*رقم|رقم\s*الصادر|الكتاب)",
        re.IGNORECASE,
    ),
]

# Generic fallback for old court cases like 2026/41.
_CASE_WITH_SEPARATOR_PATTERN = re.compile(
    r"(?<![\d/])((?:20\d{2}|19\d{2})\s*[\/\-]\s*\d{1,6})(?![\d/])"
)

_COMPACT_CASE_PATTERN = re.compile(
    r"(?<!\d)((?:20\d{2}|19\d{2})(?:\d{1,4}))(?!\d)"
)

_VALUE_AFTER_COLON = re.compile(r"[:：.]\s*(.+)$")


# ---------------------------------------------------------------------------
# Strong local date fallback patterns
# ---------------------------------------------------------------------------

_LOCAL_DATE_YMD = re.compile(
    r"(?<!\d)((?:20\d{2}|19\d{2})[\/\-.](0?[1-9]|1[0-2])[\/\-.](0?[1-9]|[12]\d|3[01]))(?!\d)"
)

_LOCAL_DATE_DMY = re.compile(
    r"(?<!\d)((0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|1[0-2])[\/\-.]((?:20\d{2}|19\d{2})))(?!\d)"
)

_LOCAL_DATE_COMPACT = re.compile(
    r"(?<!\d)((?:20\d{2}|19\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01]))(?!\d)"
)

_LOCAL_DATE_YEAR_SLASH_MMDD = re.compile(
    r"(?<!\d)((?:20\d{2}|19\d{2})[\/\-.]((?:0?[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])))(?!\d)"
)

OCR_DATE_KEYWORD_VARIANTS = [
    "التاريخ",
    "تاريخ",
    "بتاريخ",
    "الموافق",
    "التاربخ",
    "التاربح",
    "التاريح",
    "تاربخ",
    "تادىخ",
    "التادىخ",
    "تادىح",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _empty_field() -> ExtractedField:
    return ExtractedField(
        value=None,
        confidence=MISSING_CONFIDENCE,
        needs_review=True,
    )


def _field(value: Optional[str], confidence: float, review: Optional[bool] = None) -> ExtractedField:
    if not value:
        return _empty_field()

    value = str(value).strip()

    if review is None:
        review = needs_review(confidence)

    return ExtractedField(
        value=value,
        confidence=float(confidence),
        needs_review=bool(review),
    )


def _clean_text_value(value: str) -> Optional[str]:
    if not value:
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.lstrip(":：-–— ").strip()

    return value if value else None


def _find_value_on_line(line: str, keyword: str) -> Optional[str]:
    """Given a line containing keyword, pull value after keyword."""
    idx = line.find(keyword)

    if idx == -1:
        return None

    remainder = line[idx + len(keyword):].strip()

    colon_match = _VALUE_AFTER_COLON.search(remainder)

    if colon_match:
        return _clean_text_value(colon_match.group(1))

    return _clean_text_value(remainder)


def _build_search_area(lines: List[str], index: int) -> str:
    """
    Search previous + current + next line.

    Needed because OCR may read:
    2026/02/04
    التاريخ

    instead of:
    التاريخ: 2026/02/04
    """
    parts = []

    if index - 1 >= 0:
        parts.append(lines[index - 1].strip())

    parts.append(lines[index].strip())

    if index + 1 < len(lines):
        parts.append(lines[index + 1].strip())

    return " ".join(parts)


def _normalize_case_number(raw_value: str) -> Optional[str]:
    """
    Normalize case number.

    Supports:
    - 1204/2026
    - 2026/41
    - 2026 - 41
    - 202641
    """
    if not raw_value:
        return None

    value = str(raw_value).strip()
    value = re.sub(r"\s+", " ", value)

    # Direct slash/hyphen case number.
    direct_match = re.search(
        r"(?<![\d/])(?P<value>\d{1,6}\s*[\/\-]\s*\d{1,6})(?![\d/])",
        value,
    )

    if direct_match:
        number = direct_match.group("value")
        number = re.sub(r"\s+", "", number)
        number = number.replace("-", "/")
        return number

    # OCR sometimes drops the "/" entirely, leaving just "2026 1322" or
    # "1322 2026" separated by whitespace only. Only accept this when
    # one side is clearly a year (20xx/19xx) -- otherwise two random
    # numbers next to each other is too ambiguous to treat as a case
    # number.
    space_match = re.search(
        r"(?<!\d)(?P<num>\d{1,6})\s+(?P<year>20\d{2}|19\d{2})(?!\d)",
        value,
    )

    if space_match:
        return f"{space_match.group('num')}/{space_match.group('year')}"

    space_match_reversed = re.search(
        r"(?<!\d)(?P<year>20\d{2}|19\d{2})\s+(?P<num>\d{1,6})(?!\d)",
        value,
    )

    if space_match_reversed:
        return f"{space_match_reversed.group('num')}/{space_match_reversed.group('year')}"

    # Old format: 202641 -> 2026/41
    compact_match = _COMPACT_CASE_PATTERN.search(value)

    if compact_match:
        compact = compact_match.group(1)

        # Avoid treating full dates like 20260204 as case numbers.
        if len(compact) == 8:
            return None

        year = compact[:4]
        rest = compact[4:]

        if rest:
            return f"{year}/{rest}"

    return None


def _normalize_document_number(value: str) -> Optional[str]:
    """
    Normalize official document/book number.

    Example:
    UW - 2026 - 0004 -> UW-2026-0004

    Every document number in this project family uses the fixed "UW-"
    prefix. When OCR misreads that prefix as something clearly garbled
    (e.g. "0|-2026-0131" instead of "UW-2026-0131" -- "U" and "W" being
    misread as digits/pipes is common in some fonts), the "-YYYY-NNNN"
    suffix is still distinctive enough to trust, so the prefix is
    corrected to "UW" rather than the whole match being discarded.
    """
    if not value:
        return None

    value = str(value).strip()
    value = re.sub(r"\s*[-–—]\s*", "-", value)
    value = re.sub(r"\s+", "", value)
    value = value.upper()

    if re.fullmatch(r"[A-Z]{1,10}-\d{4}-\d{3,10}", value):
        return value

    # Prefix wasn't clean letters (digits/pipes/etc bled in) but the
    # "-YYYY-NNNN" suffix shape is still a confident match -- recover
    # the known-fixed "UW" prefix instead of discarding this entirely.
    loose_match = re.fullmatch(r"[A-Z0-9|/]{1,10}-(\d{4}-\d{3,10})", value)

    if loose_match:
        return f"UW-{loose_match.group(1)}"

    return None


def _extract_case_like_value(search_area: str) -> Optional[str]:
    return _normalize_case_number(search_area)


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def _normalize_date_string(raw_date: str) -> Optional[str]:
    """Normalize a raw date string into YYYY/MM/DD."""
    if not raw_date:
        return None

    value = str(raw_date).strip()
    value = value.replace("-", "/").replace(".", "/")
    value = re.sub(r"\s+", "", value)

    # 20260204 -> 2026/02/04
    if re.fullmatch(r"(?:20\d{2}|19\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])", value):
        return f"{value[0:4]}/{value[4:6]}/{value[6:8]}"

    # 2026/0204 -> 2026/02/04
    match = re.fullmatch(
        r"((?:20\d{2}|19\d{2}))\/((?:0?[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))",
        value,
    )

    if match:
        year = match.group(1)
        rest = match.group(2).zfill(4)
        return f"{year}/{rest[0:2]}/{rest[2:4]}"

    # 2026/02/04 or 05/07/2026
    parts = value.split("/")

    if len(parts) == 3:
        first, second, third = parts

        # YYYY/MM/DD
        if len(first) == 4:
            return f"{first}/{second.zfill(2)}/{third.zfill(2)}"

        # DD/MM/YYYY
        if len(third) == 4:
            return f"{third}/{second.zfill(2)}/{first.zfill(2)}"

    return None


def _normalize_date_value(text: str) -> Optional[str]:
    """
    Find a date anywhere in text and normalize it to YYYY/MM/DD.

    Supports:
    - 2026/02/04
    - 2026-02-04
    - 04/02/2026
    - 20260204
    - 2026/0204
    """
    if not text:
        return None

    # OCR sometimes inserts spaces around date separators (e.g.
    # "2026 / 08 / 02" instead of "2026/08/02"). None of the patterns
    # below tolerate that, so collapse it first -- only between digits,
    # so this never touches unrelated punctuation elsewhere in the text.
    text = re.sub(r"(?<=\d)\s*([\/\-.])\s*(?=\d)", r"\1", text)

    match = DATE_PATTERN_YMD.search(text)

    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}/{int(month):02d}/{int(day):02d}"

    match = DATE_PATTERN_DMY.search(text)

    if match:
        day, month, year = match.group(1), match.group(2), match.group(3)
        return f"{year}/{int(month):02d}/{int(day):02d}"

    match = DATE_PATTERN_COMPACT.search(text)

    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}/{int(month):02d}/{int(day):02d}"

    match = _LOCAL_DATE_YMD.search(text)

    if match:
        return _normalize_date_string(match.group(1))

    match = _LOCAL_DATE_DMY.search(text)

    if match:
        return _normalize_date_string(match.group(1))

    match = _LOCAL_DATE_COMPACT.search(text)

    if match:
        return _normalize_date_string(match.group(1))

    match = _LOCAL_DATE_YEAR_SLASH_MMDD.search(text)

    if match:
        return _normalize_date_string(match.group(1))

    return None


def _extract_date_value(search_area: str) -> Optional[str]:
    return _normalize_date_value(search_area)


def _extract_first_date_anywhere(text: Optional[str]) -> Optional[str]:
    """
    Strong fallback:
    If header OCR contains a date, extract it even if the keyword is missing
    or OCR read "التاريخ" incorrectly as "التاربخ".

    Document/book numbers (e.g. "UW-2026-0131") are stripped out first --
    their "-YYYY-NNNN" shape can otherwise be misread by the date
    patterns as a compact date (e.g. "2026/01/31"), especially when the
    real date's own keyword was too garbled by OCR to match directly.
    """
    if not text:
        return None

    text_without_document_numbers = text

    for pattern in DOCUMENT_NUMBER_PATTERNS:
        text_without_document_numbers = pattern.sub(" ", text_without_document_numbers)

    # Also strip bare "-YYYY-NNNN" shapes even without a nearby keyword,
    # in case the document number appears without "رقم الكتاب" nearby in
    # this particular text window.
    text_without_document_numbers = re.sub(
        r"[A-Za-z0-9|]{1,10}\s*[-–—]\s*\d{4}\s*[-–—]\s*\d{3,10}",
        " ",
        text_without_document_numbers,
    )

    return _normalize_date_value(text_without_document_numbers)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

MAX_COURT_NAME_LENGTH = 60

_COURT_NAME_REGEX = re.compile(
    r"محكمة\s+[\u0600-\u06FF]{2,20}(?:\s+[\u0600-\u06FF]{2,20}){0,4}"
)


def _normalize_spaces_local(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_KNOWN_COURT_WORD_PATTERN = re.compile(r"[\u0600-\u06FF]{2,}")


def _fuzzy_word_match(word_a: str, word_b: str) -> bool:
    if word_a == word_b:
        return True
    return difflib.SequenceMatcher(None, word_a, word_b).ratio() >= 0.72


def _match_known_court_name(search_text: str) -> Optional[str]:
    """
    Fuzzy-match search_text against the closed, known list of Jordanian
    court names (utils/known_courts.py) and return the correct,
    canonically-spelled name if a confident match is found.

    Unlike person names -- which are open-ended and can't safely be
    "corrected" against a dictionary -- court names come from a small
    fixed set defined by the judiciary. That makes it safe to recover
    the correct name (city included) even when OCR misspells a word
    (e.g. "بدابة" instead of "بداية") or splits it away from the rest
    of the line.

    Matching is done in a LOCAL window right after each occurrence of
    "محكمة" in the text (not the whole document) -- a wide search window
    can accidentally combine a court-type word from one part of the
    document (e.g. "بداية" from the unrelated legal phrase "بداية حقوق")
    with a city name mentioned somewhere else entirely, producing a
    false match. Keeping matches local to one actual "محكمة" mention
    avoids that.
    """
    if not search_text:
        return None

    best_court = None
    best_word_count = 0
    best_position = None

    for occurrence in re.finditer("محكمة", search_text):
        window = search_text[occurrence.start(): occurrence.start() + 80]
        window_words = _KNOWN_COURT_WORD_PATTERN.findall(window)[:4]

        if not window_words:
            continue

        for court in KNOWN_COURT_NAMES:
            court_words = court.split()

            if all(
                any(_fuzzy_word_match(cw, ww) for ww in window_words)
                for cw in court_words
            ):
                is_better = len(court_words) > best_word_count or (
                    len(court_words) == best_word_count
                    and (best_position is None or occurrence.start() < best_position)
                )

                if is_better:
                    best_word_count = len(court_words)
                    best_court = court
                    best_position = occurrence.start()

    return best_court


def _extract_court_name(lines: List[str]) -> ExtractedField:
    """
    Court name is usually in document header.
    Example: محكمة بداية العقبة

    Requires at least one Arabic word AFTER "محكمة" before accepting a
    match -- this avoids grabbing an isolated "محكمة" token with nothing
    after it (common when the cropped header OCR is garbled).
    """
    for line in lines[:25]:
        clean_line = line.strip()
        match = _COURT_NAME_REGEX.search(clean_line)

        if match:
            value = _normalize_spaces_local(match.group())

            if len(value) <= MAX_COURT_NAME_LENGTH:
                return _field(value, EXACT_MATCH_CONFIDENCE, False)

    for i in range(min(len(lines) - 1, 25)):
        joined = f"{lines[i].strip()} {lines[i + 1].strip()}"
        match = _COURT_NAME_REGEX.search(joined)

        if match:
            value = _normalize_spaces_local(match.group())

            if len(value) <= MAX_COURT_NAME_LENGTH:
                return _field(value, NEARBY_MATCH_CONFIDENCE, False)

    for line in lines:
        clean_line = line.strip()

        for keyword in COURT_NAME_KEYWORDS:
            if keyword in clean_line and len(clean_line) <= MAX_COURT_NAME_LENGTH:
                return _field(clean_line, NEARBY_MATCH_CONFIDENCE, False)

    return _empty_field()


def _extract_case_number(lines: List[str]) -> ExtractedField:
    """
    Extract case number.

    Expected examples:
    - رقم القضية : 1204/2026
    - بالدعوى رقم 2026/41
    """
    full_text = "\n".join(lines)

    for pattern in CASE_NUMBER_PATTERNS:
        match = pattern.search(full_text)

        if match:
            value = _normalize_case_number(match.group("value"))

            if value:
                return _field(value, 0.90, False)

    for i, line in enumerate(lines):
        for keyword in CASE_NUMBER_KEYWORDS:
            if keyword in line:
                value_after_keyword = _find_value_on_line(line, keyword)
                value = _extract_case_like_value(value_after_keyword or line)

                if not value and i + 1 < len(lines):
                    value = _extract_case_like_value(lines[i + 1])

                if value:
                    return _field(value, EXACT_MATCH_CONFIDENCE, False)

                return _field(None, NEARBY_MATCH_CONFIDENCE, True)

    for line in lines:
        if any(keyword in line for keyword in DOCUMENT_NUMBER_KEYWORDS):
            continue

        if "الكتاب" in line:
            continue

        if _normalize_date_value(line):
            continue

        value = _extract_case_like_value(line)

        if value:
            return _field(value, NEARBY_MATCH_CONFIDENCE, True)

    return _empty_field()


def _extract_document_number(lines: List[str]) -> ExtractedField:
    """
    Extract official document/book number.

    Expected examples:
    - رقم الكتاب : UW-2026-0004
    - UW-2026-0004 : رقم الكتاب

    Important:
    Do not fallback to case number here.
    If book number is not found, return empty and require review.
    """
    header_lines = lines[:25]
    header_text = "\n".join(header_lines)

    for pattern in DOCUMENT_NUMBER_PATTERNS:
        match = pattern.search(header_text)

        if match:
            value = _normalize_document_number(match.group("value"))

            if value:
                return _field(value, 0.90, False)

    full_text = "\n".join(lines)

    for pattern in DOCUMENT_NUMBER_PATTERNS:
        match = pattern.search(full_text)

        if match:
            value = _normalize_document_number(match.group("value"))

            if value:
                return _field(value, 0.90, False)

    return _empty_field()


def _extract_document_date(lines: List[str]) -> ExtractedField:
    """
    Extract date from:
    التاريخ: 2026/02/04

    Also supports OCR cases:
    2026/02/04
    التاربخ
    """
    for i, line in enumerate(lines):
        all_date_keywords = list(DOCUMENT_DATE_KEYWORDS) + OCR_DATE_KEYWORD_VARIANTS

        for keyword in all_date_keywords:
            if keyword in line:
                search_area = _build_search_area(lines, i)
                value = _extract_date_value(search_area)

                if value:
                    return _field(value, EXACT_MATCH_CONFIDENCE, False)

                return _field(None, NEARBY_MATCH_CONFIDENCE, True)

    full_text = "\n".join(lines)
    value = _extract_first_date_anywhere(full_text)

    if value:
        return _field(value, 0.85, False)

    return _empty_field()


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_document_fields(cleaned_text: str, header_text: Optional[str] = None) -> DocumentInfo:
    """Run document-level extraction over normalized OCR text.

    header_text is searched first for document_date and header fields because
    the header crop is the most reliable place for:
    - court name
    - case number
    - document/book number
    - document date
    """
    if cleaned_text is None:
        cleaned_text = ""

    if header_text is None:
        header_text = ""

    combined_text = f"{header_text}\n{cleaned_text}".strip()
    lines = [line.strip() for line in combined_text.splitlines() if line.strip()]

    court_search_text = f"{header_text}\n{cleaned_text[:1500]}"
    matched_court = _match_known_court_name(court_search_text)

    if matched_court:
        court_name = _field(matched_court, EXACT_MATCH_CONFIDENCE, False)
    else:
        court_name = _extract_court_name(lines)

    case_number = _extract_case_number(lines)
    document_number = _extract_document_number(lines)

    document_date = _empty_field()

    if header_text:
        direct_header_date = _extract_first_date_anywhere(header_text)

        if direct_header_date:
            document_date = _field(direct_header_date, 0.90, False)
        else:
            header_lines = [line.strip() for line in header_text.splitlines() if line.strip()]
            document_date = _extract_document_date(header_lines)

    if document_date.value is None:
        direct_text_date = _extract_first_date_anywhere(cleaned_text)

        if direct_text_date:
            document_date = _field(direct_text_date, 0.85, False)
        else:
            document_date = _extract_document_date(lines)

    return DocumentInfo(
        court_name=court_name,
        case_number=case_number,
        document_number=document_number,
        document_date=document_date,
    )