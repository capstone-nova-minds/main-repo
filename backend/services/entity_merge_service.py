"""Merge rule-based person/company candidates with local NER entities.

Rule-based extraction stays the source of truth; NER only fills gaps
(names the keyword rules missed) and helps confirm company vs
individual. Every NER-only addition is marked needs_review=true since
it has no keyword/National-ID confirmation behind it.
"""

import re
from typing import Any, Dict, List

from schemas.person_schema import PersonRecord
from utils.regex_patterns import NATIONAL_ID_PATTERN, COMPANY_KEYWORDS
from services.ner_service import PERSON_LABELS, ORG_LABELS

PROXIMITY_WINDOW = 60  # characters to look around a NER entity for a nearby National ID
NER_DEFAULT_CONFIDENCE = 0.75


def _normalize_for_match(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip()


def _is_duplicate_name(name: str, existing_names: List[str]) -> bool:
    """Simple Arabic name de-duplication: exact match or one contains the other."""
    normalized = _normalize_for_match(name)
    if not normalized:
        return False
    for existing in existing_names:
        existing_normalized = _normalize_for_match(existing)
        if not existing_normalized:
            continue
        if normalized == existing_normalized:
            return True
        if normalized in existing_normalized or existing_normalized in normalized:
            return True
    return False


def _find_nearby_national_id(text: str, start_char, end_char) -> str:
    if start_char is None or end_char is None:
        return None
    window_start = max(0, start_char - PROXIMITY_WINDOW)
    window_end = min(len(text), end_char + PROXIMITY_WINDOW)
    surrounding = text[window_start:window_end]
    match = NATIONAL_ID_PATTERN.search(surrounding)
    return match.group(0) if match else None


def _surrounding_has_company_keyword(text: str, start_char, end_char) -> bool:
    if start_char is None or end_char is None:
        return False
    window_start = max(0, start_char - PROXIMITY_WINDOW)
    window_end = min(len(text), end_char + PROXIMITY_WINDOW)
    surrounding = text[window_start:window_end]
    return any(keyword in surrounding for keyword in COMPANY_KEYWORDS)


def merge_rules_and_ner(
    cleaned_text: str,
    rule_based_records: List[PersonRecord],
    ner_result: Dict[str, Any],
) -> List[PersonRecord]:
    """Combine rule-based candidates with NER entities into a final list."""
    merged: List[PersonRecord] = list(rule_based_records)
    existing_names = [r.full_name for r in merged if r.full_name]

    if ner_result.get("ner_status") != "success":
        # NER unavailable/failed -- continue with rules-only results.
        return merged

    for entity in ner_result.get("entities", []):
        label = (entity.get("label") or "").upper()
        text = entity.get("text", "")
        if not text:
            continue

        start_char = entity.get("start_char")
        end_char = entity.get("end_char")

        if label in PERSON_LABELS:
            if _is_duplicate_name(text, existing_names):
                continue

            nearby_id = _find_nearby_national_id(cleaned_text, start_char, end_char)
            is_company_context = _surrounding_has_company_keyword(cleaned_text, start_char, end_char)

            record = PersonRecord(
                full_name=text,
                national_id=nearby_id,
                registration_number=None,
                person_type="Company" if is_company_context else "Individual",
                confidence=entity.get("confidence", NER_DEFAULT_CONFIDENCE),
                needs_review=nearby_id is None,  # no National ID confirmation -> review
                source="ner",
            )
            merged.append(record)
            existing_names.append(text)

        elif label in ORG_LABELS:
            if _is_duplicate_name(text, existing_names):
                # Same org already present from rules -- just confirm it's a Company.
                for r in merged:
                    if _is_duplicate_name(r.full_name or "", [text]):
                        r.person_type = "Company"
                continue

            record = PersonRecord(
                full_name=text,
                national_id=None,
                registration_number=None,
                person_type="Company",
                confidence=entity.get("confidence", NER_DEFAULT_CONFIDENCE),
                needs_review=True,
                source="ner",
            )
            merged.append(record)
            existing_names.append(text)

    return merged
