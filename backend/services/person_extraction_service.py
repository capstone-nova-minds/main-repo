"""Rule-based extraction of person/company candidate records.

Order of operations (also mirrored by entity_merge_service, which adds
NER on top of this):
  1. Find National IDs via regex.
  2. Find candidate lines via Arabic legal keywords.
  3. Detect company vs individual.
  4. Attach a nearby National ID to a candidate line if present.
"""

import re
from typing import List, Optional

from schemas.person_schema import PersonRecord
from utils.confidence import EXACT_MATCH_CONFIDENCE, NEARBY_MATCH_CONFIDENCE, WEAK_MATCH_CONFIDENCE
from utils.regex_patterns import NATIONAL_ID_PATTERN, PERSON_LINE_KEYWORDS, COMPANY_KEYWORDS

# Strip the keyword itself and leading punctuation/labels from a candidate line.
_LEADING_NOISE = re.compile(r"^[:\-\s،]+")


def is_company_line(line: str) -> bool:
    return any(keyword in line for keyword in COMPANY_KEYWORDS)


def _clean_name(line: str, keyword: str) -> str:
    """Remove the triggering keyword and the National ID from a candidate line."""
    without_keyword = line.replace(keyword, " ")
    without_id = NATIONAL_ID_PATTERN.sub(" ", without_keyword)
    name = _LEADING_NOISE.sub("", without_id).strip(" :-،")
    return re.sub(r"\s+", " ", name).strip()


def find_national_ids(text: str) -> List[str]:
    """All valid 11-digit National IDs found in the (already normalized) text."""
    return NATIONAL_ID_PATTERN.findall(text)


def extract_candidate_lines(cleaned_text: str) -> List[str]:
    """Lines that mention a person/company legal keyword."""
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if any(keyword in line for keyword in PERSON_LINE_KEYWORDS):
            candidates.append(line)
    return candidates


def _matched_keyword(line: str) -> Optional[str]:
    for keyword in PERSON_LINE_KEYWORDS:
        if keyword in line:
            return keyword
    return None


def extract_person_candidates(cleaned_text: str) -> List[PersonRecord]:
    """Build PersonRecord candidates from rule-based line matching alone.

    NER-derived candidates are added later by entity_merge_service.
    """
    candidate_lines = extract_candidate_lines(cleaned_text)
    records: List[PersonRecord] = []

    for line in candidate_lines:
        keyword = _matched_keyword(line)
        if keyword is None:
            continue

        national_ids = find_national_ids(line)
        national_id = national_ids[0] if national_ids else None
        name = _clean_name(line, keyword)
        is_company = is_company_line(line)

        if not name:
            # Nothing usable to build a record from.
            continue

        if is_company:
            record = PersonRecord(
                full_name=name,
                national_id=None,
                registration_number=None,
                person_type="Company",
                confidence=NEARBY_MATCH_CONFIDENCE,
                needs_review=True,
                source="rules",
            )
        elif national_id:
            record = PersonRecord(
                full_name=name,
                national_id=national_id,
                registration_number=None,
                person_type="Individual",
                confidence=EXACT_MATCH_CONFIDENCE,
                needs_review=False,
                source="rules",
            )
        else:
            record = PersonRecord(
                full_name=name,
                national_id=None,
                registration_number=None,
                person_type="Individual",
                confidence=WEAK_MATCH_CONFIDENCE,
                needs_review=True,
                source="rules",
            )

        records.append(record)

    return records
