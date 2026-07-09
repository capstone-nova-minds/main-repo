"""Rule-based extraction of document-level fields (court, case number,
document number, date) using deterministic keyword + regex matching.

No LLM, no ML model -- every decision here is explainable.
"""

import re
from typing import Dict, List, Optional

from schemas.document_schema import DocumentInfo, ExtractedField
from utils.confidence import (
    EXACT_MATCH_CONFIDENCE,
    NEARBY_MATCH_CONFIDENCE,
    MISSING_CONFIDENCE,
    needs_review,
)
from utils.regex_patterns import (
    COURT_NAME_KEYWORDS,
    CASE_NUMBER_KEYWORDS,
    DOCUMENT_NUMBER_KEYWORDS,
    DOCUMENT_DATE_KEYWORDS,
    DATE_PATTERN,
)

# Matches "<keyword> ... : <value>" or "<keyword> ... <value>" on one line.
_VALUE_AFTER_COLON = re.compile(r"[:：]\s*(.+)$")
_TRAILING_NUMBER_OR_TEXT = re.compile(r"([\w/\-؀-ۿ]+)\s*$")


def _empty_field() -> ExtractedField:
    return ExtractedField(value=None, confidence=MISSING_CONFIDENCE, needs_review=True)


def _find_value_on_line(line: str, keyword: str) -> Optional[str]:
    """Given a line containing `keyword`, try to pull the value after it."""
    idx = line.find(keyword)
    if idx == -1:
        return None
    remainder = line[idx + len(keyword):].strip()

    colon_match = _VALUE_AFTER_COLON.search(remainder)
    if colon_match:
        value = colon_match.group(1).strip()
        return value if value else None

    remainder = remainder.lstrip(":：-  ").strip()
    return remainder if remainder else None


def _extract_court_name(lines: List[str]) -> ExtractedField:
    for line in lines:
        for keyword in COURT_NAME_KEYWORDS:
            if keyword in line:
                # Court name is usually the whole line itself.
                return ExtractedField(
                    value=line.strip(),
                    confidence=EXACT_MATCH_CONFIDENCE,
                    needs_review=needs_review(EXACT_MATCH_CONFIDENCE),
                )
    return _empty_field()


def _extract_keyword_value(lines: List[str], keywords: List[str]) -> ExtractedField:
    for line in lines:
        for keyword in keywords:
            if keyword in line:
                value = _find_value_on_line(line, keyword)
                if value:
                    return ExtractedField(
                        value=value,
                        confidence=EXACT_MATCH_CONFIDENCE,
                        needs_review=needs_review(EXACT_MATCH_CONFIDENCE),
                    )
                return ExtractedField(
                    value=None,
                    confidence=NEARBY_MATCH_CONFIDENCE,
                    needs_review=True,
                )
    return _empty_field()


def _extract_document_date(lines: List[str]) -> ExtractedField:
    for line in lines:
        for keyword in DOCUMENT_DATE_KEYWORDS:
            if keyword in line:
                date_match = DATE_PATTERN.search(line)
                if date_match:
                    return ExtractedField(
                        value=date_match.group(0),
                        confidence=EXACT_MATCH_CONFIDENCE,
                        needs_review=needs_review(EXACT_MATCH_CONFIDENCE),
                    )
    # Fall back: any recognizable date pattern anywhere in the text.
    for line in lines:
        date_match = DATE_PATTERN.search(line)
        if date_match:
            return ExtractedField(
                value=date_match.group(0),
                confidence=NEARBY_MATCH_CONFIDENCE,
                needs_review=True,
            )
    return _empty_field()


def extract_document_fields(cleaned_text: str) -> DocumentInfo:
    """Run all document-level extraction rules over normalized OCR text."""
    lines = [line for line in cleaned_text.splitlines() if line.strip()]

    return DocumentInfo(
        court_name=_extract_court_name(lines),
        case_number=_extract_keyword_value(lines, CASE_NUMBER_KEYWORDS),
        document_number=_extract_keyword_value(lines, DOCUMENT_NUMBER_KEYWORDS),
        document_date=_extract_document_date(lines),
    )
