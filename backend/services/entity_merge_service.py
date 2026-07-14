"""Merge rule-based person/company candidates with local NER entities.

Rule-based extraction stays the source of truth.
NER is only used to fill gaps, not to override better rule-based records.

Main rules:
- If two records have the same National ID, keep the better one.
- Prefer rule-based records over NER.
- Prefer longer full names.
- NER PERSON without nearby National ID is NOT added as final person.
- NER PERSON without ID can still be used to enrich a short rule-based name.
- ORG entities can become Company records.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from schemas.person_schema import PersonRecord
from utils.regex_patterns import NATIONAL_ID_PATTERN, COMPANY_KEYWORDS
from services.ner_service import PERSON_LABELS, ORG_LABELS


PROXIMITY_WINDOW = 120
NER_DEFAULT_CONFIDENCE = 0.75

_ARABIC_WORD_PATTERN = re.compile(r"[\u0600-\u06FF]{2,}")


def _normalize_for_match(name: str) -> str:
    """Normalize name for duplicate matching."""
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


def _find_nearby_national_id(text: str, start_char, end_char) -> Optional[str]:
    """Find National ID near a NER entity."""
    if start_char is None or end_char is None:
        return None

    window_start = max(0, start_char - PROXIMITY_WINDOW)
    window_end = min(len(text), end_char + PROXIMITY_WINDOW)
    surrounding = text[window_start:window_end]

    match = NATIONAL_ID_PATTERN.search(surrounding)

    return match.group(0) if match else None


def _surrounding_has_company_keyword(text: str, start_char, end_char) -> bool:
    """Check if an entity appears near company keywords."""
    if start_char is None or end_char is None:
        return False

    window_start = max(0, start_char - PROXIMITY_WINDOW)
    window_end = min(len(text), end_char + PROXIMITY_WINDOW)
    surrounding = text[window_start:window_end]

    return any(keyword in surrounding for keyword in COMPANY_KEYWORDS)


def _name_word_count(record: PersonRecord) -> int:
    """Count words in full_name."""
    if not record.full_name:
        return 0

    return len(record.full_name.split())


def _record_score(record: PersonRecord) -> int:
    """Score record quality. Higher score wins during merge."""
    score = 0

    if record.national_id:
        score += 10

    if record.source == "rules":
        score += 8

    if getattr(record, "extraction_method", None) == "direct_legal_phrase":
        score += 4

    score += _name_word_count(record)

    if record.needs_review is False:
        score += 2

    return score


def _copy_person_with_updates(person: PersonRecord, **updates) -> PersonRecord:
    """Create updated PersonRecord while supporting Pydantic v1/v2."""
    if hasattr(person, "model_dump"):
        data = person.model_dump()
    else:
        data = person.dict()

    data.update(updates)

    return PersonRecord(**data)


def _merge_same_key_records(existing: PersonRecord, new: PersonRecord) -> PersonRecord:
    """Merge two records that represent the same person/company."""
    existing_score = _record_score(existing)
    new_score = _record_score(new)

    best = new if new_score > existing_score else existing
    other = existing if best is new else new

    update_data = {}

    if not best.national_id and other.national_id:
        update_data["national_id"] = other.national_id

    if not best.registration_number and other.registration_number:
        update_data["registration_number"] = other.registration_number

    if not getattr(best, "extraction_method", None) and getattr(other, "extraction_method", None):
        update_data["extraction_method"] = other.extraction_method

    # Use whichever record scored higher (see _record_score) for
    # person_type, rather than "if either says Company, it's Company".
    # That OR-logic let a low-confidence NER guess (e.g. fooled by a
    # nearby company's keywords when several people are listed close
    # together) override a reliable rules-based "Individual"
    # classification -- which also caused a valid national_id to be
    # incorrectly stripped downstream (companies can't have one).
    if best.person_type != other.person_type:
        update_data["person_type"] = best.person_type

    if best.national_id and best.full_name and len(best.full_name.split()) >= 3:
        update_data["needs_review"] = False

    if existing.source != new.source:
        if "rules" in [existing.source, new.source]:
            update_data["source"] = "rules+ner"

    if update_data:
        best = _copy_person_with_updates(best, **update_data)

    return best


def _deduplicate_and_prefer_best(records: List[PersonRecord]) -> List[PersonRecord]:
    """
    Remove duplicates.

    Priority:
    1. Same national_id.
    2. Duplicate/similar name.
    3. Prefer rules.
    4. Prefer longer full_name.
    """
    best_by_id = {}
    no_id_records = []

    for record in records:
        if record.national_id:
            key = record.national_id

            if key not in best_by_id:
                best_by_id[key] = record
            else:
                best_by_id[key] = _merge_same_key_records(best_by_id[key], record)
        else:
            no_id_records.append(record)

    merged = list(best_by_id.values())

    for record in no_id_records:
        existing_match_index = None

        for idx, existing in enumerate(merged):
            if _is_duplicate_name(record.full_name or "", [existing.full_name or ""]):
                existing_match_index = idx
                break

        if existing_match_index is not None:
            merged[existing_match_index] = _merge_same_key_records(
                merged[existing_match_index],
                record,
            )
        else:
            merged.append(record)

    return merged


def _get_person_ner_texts(ner_result: Dict[str, Any]) -> List[str]:
    """Return PER/PERSON entity texts from NER result."""
    entities = ner_result.get("entities", []) if ner_result else []
    person_texts = []

    for entity in entities:
        label = str(entity.get("label", "")).upper()
        text = str(entity.get("text", "")).strip()

        if label in PERSON_LABELS and text:
            words = _ARABIC_WORD_PATTERN.findall(text)

            if len(words) >= 2:
                person_texts.append(" ".join(words))

    return person_texts


def _combine_ner_with_short_rule_name(rule_name: str, ner_text: str) -> Optional[str]:
    """
    Combine short rule-based name with NER text.

    Example:
    rule_name = يوسف الخوالدة
    ner_text = فواز سامي

    Output:
    سامي فواز يوسف الخوالدة
    """
    rule_words = _ARABIC_WORD_PATTERN.findall(rule_name or "")
    ner_words = _ARABIC_WORD_PATTERN.findall(ner_text or "")

    if len(rule_words) < 2 or len(ner_words) < 2:
        return None

    # In this sample, NER returns "فواز سامي",
    # but expected order is "سامي فواز".
    ner_words_reversed = list(reversed(ner_words))

    combined_words = []

    for word in ner_words_reversed + rule_words:
        if word not in combined_words:
            combined_words.append(word)

    if len(combined_words) < 4:
        return None

    return " ".join(combined_words)


def _enrich_short_rule_names_with_ner(
    merged_persons: List[PersonRecord],
    ner_result: Dict[str, Any],
) -> List[PersonRecord]:
    """
    If a rule-based person has National ID but only 2 words, try to
    enrich it using an NER PERSON suggestion that has no nearby ID.

    Only enriches in the unambiguous 1-to-1 case: exactly one person
    needs enrichment AND exactly one NER suggestion is available. With
    more than one candidate on either side there's no reliable way to
    know which suggestion belongs to which person -- guessing previously
    caused the SAME enriched name to be applied to multiple different
    people. In the ambiguous case, the original short rule-based name is
    kept (still needs_review=True) instead of fabricating a name.
    """
    person_ner_texts = _get_person_ner_texts(ner_result)

    if not person_ner_texts:
        return merged_persons

    enrichable_indexes = []

    for idx, person in enumerate(merged_persons):
        full_name = getattr(person, "full_name", "") or ""
        national_id = getattr(person, "national_id", None)
        person_type = getattr(person, "person_type", "Individual")
        words = full_name.split()

        if person_type == "Individual" and national_id and len(words) <= 2:
            enrichable_indexes.append(idx)

    if len(enrichable_indexes) != 1 or len(person_ner_texts) != 1:
        return merged_persons

    target_idx = enrichable_indexes[0]
    person = merged_persons[target_idx]
    full_name = getattr(person, "full_name", "") or ""

    best_name = _combine_ner_with_short_rule_name(full_name, person_ner_texts[0])

    if not best_name:
        return merged_persons

    enriched_person = _copy_person_with_updates(
        person,
        full_name=best_name,
        confidence=max(float(getattr(person, "confidence", 0.0)), 0.75),
        needs_review=True,
        source="rules+ner",
        extraction_method="ner_enriched_national_id_context",
    )

    result = list(merged_persons)
    result[target_idx] = enriched_person
    return result


def merge_rules_and_ner(
    cleaned_text: str,
    rule_based_records: List[PersonRecord],
    ner_result: Dict[str, Any],
) -> Tuple[List[PersonRecord], List[Dict[str, Any]]]:
    """
    Combine rule-based candidates with NER entities into a final list.

    Returns:
    - persons
    - suggested_entities
    """
    merged: List[PersonRecord] = list(rule_based_records)
    suggested_entities: List[Dict[str, Any]] = []

    if ner_result.get("ner_status") != "success":
        merged = _deduplicate_and_prefer_best(merged)
        return merged, suggested_entities

    for entity in ner_result.get("entities", []):
        label = (entity.get("label") or "").upper()
        entity_text = _normalize_for_match(entity.get("text", ""))

        if not entity_text:
            continue

        start_char = entity.get("start_char")
        end_char = entity.get("end_char")
        entity_confidence = float(entity.get("confidence", NER_DEFAULT_CONFIDENCE))

        if label in PERSON_LABELS:
            nearby_id = _find_nearby_national_id(cleaned_text, start_char, end_char)
            is_company_context = _surrounding_has_company_keyword(
                cleaned_text,
                start_char,
                end_char,
            )

            if nearby_id is None:
                suggested_entities.append(
                    {
                        "text": entity_text,
                        "label": label,
                        "confidence": entity_confidence,
                        "reason": "no_nearby_national_id",
                    }
                )
                continue

            record = PersonRecord(
                full_name=entity_text,
                national_id=nearby_id,
                registration_number=None,
                person_type="Company" if is_company_context else "Individual",
                confidence=entity_confidence,
                needs_review=False,
                source="ner",
                extraction_method="nearby_ner_person",
            )

            merged.append(record)

        elif label in ORG_LABELS:
            record = PersonRecord(
                full_name=entity_text,
                national_id=None,
                registration_number=None,
                person_type="Company",
                confidence=entity_confidence,
                needs_review=True,
                source="ner",
                extraction_method="ner_org",
            )

            merged.append(record)

    # First deduplicate normal records.
    merged = _deduplicate_and_prefer_best(merged)

    # Then enrich short rule names using NER suggestions.
    merged = _enrich_short_rule_names_with_ner(merged, ner_result)

    # Deduplicate again after enrichment.
    merged = _deduplicate_and_prefer_best(merged)

    return merged, suggested_entities