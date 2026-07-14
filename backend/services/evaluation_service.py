"""Compare an extracted/reviewed result against a hand-labeled expected
JSON, field by field, so accuracy improvements can be tracked over time.

Expected JSON shape (see tests/expected_outputs/sample_1_expected.json):
{
  "document": {
    "court_name": "...", "case_number": "...",
    "document_number": "...", "document_date": "..."
  },
  "persons": [
    {"full_name": "...", "national_id": "...", "person_type": "..."}
  ]
}

Only the first expected person is compared against the first actual
person -- this is a simple single-person evaluation, not a general
set-matching algorithm (documents in this system typically list one
primary respondent).
"""

import re
from typing import Any, Dict, List, Optional

DOCUMENT_FIELDS = ["court_name", "case_number", "document_number", "document_date"]
PERSON_FIELDS = ["full_name", "national_id", "person_type"]


def _get_document_value(document: Dict[str, Any], field_name: str) -> Optional[str]:
    """Actual results store document fields as {"value": ..., ...}; expected
    fixtures store them as plain strings. Support both shapes."""
    field = document.get(field_name)
    if isinstance(field, dict):
        return field.get("value")
    return field


def _get_first_person(result: Dict[str, Any]) -> Dict[str, Any]:
    persons = result.get("persons") or []
    return persons[0] if persons else {}


def _values_match(expected_value: Optional[str], actual_value: Optional[str]) -> bool:
    if expected_value is None:
        return actual_value is None
    if actual_value is None:
        return False
    return str(expected_value).strip() == str(actual_value).strip()


def evaluate_accuracy(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Compare `actual` (extracted/reviewed result) against `expected` (fixture).

    Returns {"total_fields", "correct_fields", "accuracy", "field_results"}.
    """
    expected_document = expected.get("document", {})
    actual_document = actual.get("document", {})

    expected_person = expected.get("persons", [{}])[0] if expected.get("persons") else {}
    actual_person = _get_first_person(actual)

    field_results: Dict[str, bool] = {}

    for field_name in DOCUMENT_FIELDS:
        expected_value = _get_document_value(expected_document, field_name)
        actual_value = _get_document_value(actual_document, field_name)
        field_results[field_name] = _values_match(expected_value, actual_value)

    for field_name in PERSON_FIELDS:
        result_key = "person_full_name" if field_name == "full_name" else field_name
        expected_value = expected_person.get(field_name)
        actual_value = actual_person.get(field_name)
        field_results[result_key] = _values_match(expected_value, actual_value)

    total_fields = len(field_results)
    correct_fields = sum(1 for is_correct in field_results.values() if is_correct)
    accuracy = round(correct_fields / total_fields, 3) if total_fields else 0.0

    return {
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "accuracy": accuracy,
        "field_results": field_results,
    }


# ---------------------------------------------------------------------------
# calculate_field_accuracy -- "Measured Field Accuracy" shown on the Review &
# Validation page: auto-extracted result vs. the same document's own human
# review (not an external hand-labeled fixture like evaluate_accuracy above).
# ---------------------------------------------------------------------------

# --- Arabic-Indic / Extended Arabic-Indic digits -> ASCII digits ---
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩"
    "۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
# Arabic diacritics (tashkeel) + superscript alef -- stripped so OCR/typing
# variants of the same word (e.g. with or without shadda) still match.
_ARABIC_DIACRITICS = re.compile("[ً-ٰٟۖ-ۭ]")
_SPACED_HYPHEN = re.compile(r"\s*-\s*")
_WHITESPACE = re.compile(r"\s+")

_PERSON_FIELD_LABELS = {
    "full_name": "Full Name",
    "national_id": "National ID",
    "registration_number": "Registration Number",
    "person_type": "Person Type",
}


def _normalize_value(value: Optional[Any]) -> Optional[str]:
    """Normalize a value for comparison: digit script, diacritics, and
    hyphen/whitespace formatting differences shouldn't count as errors."""
    if value is None:
        return None
    text = str(value).translate(_DIGIT_TRANSLATION)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _SPACED_HYPHEN.sub("-", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def _normalize_date_value(value: Optional[Any]) -> Optional[str]:
    """Normalize a date to YYYY-MM-DD regardless of whether it was written
    YYYY/MM/DD or DD/MM/YYYY, so reviewers re-typing the same date in a
    different format aren't penalized."""
    if value is None:
        return None
    text = str(value).translate(_DIGIT_TRANSLATION).strip()
    parts = re.split(r"[/-]", text)
    if len(parts) != 3:
        return _normalize_value(text)

    if len(parts[0]) == 4:
        year, month, day = parts
    elif len(parts[2]) == 4:
        day, month, year = parts
    else:
        return _normalize_value(text)

    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return _normalize_value(text)


def _values_equal(auto_value: Optional[Any], reviewed_value: Optional[Any], normalizer) -> bool:
    auto_norm = normalizer(auto_value)
    reviewed_norm = normalizer(reviewed_value)
    if auto_norm is None and reviewed_norm is None:
        return True
    return auto_norm == reviewed_norm


def _person_field_defs(person_type: Optional[str]):
    identifier_key = "registration_number" if person_type == "Company" else "national_id"
    return [
        ("full_name", _normalize_value),
        (identifier_key, _normalize_value),
        ("person_type", _normalize_value),
    ]


def _add_field_result(
    field_results: List[Dict[str, Any]],
    label: str,
    auto_value: Optional[Any],
    reviewed_value: Optional[Any],
    normalizer,
) -> None:
    field_results.append(
        {
            "field": label,
            "auto_value": auto_value,
            "reviewed_value": reviewed_value,
            "correct": _values_equal(auto_value, reviewed_value, normalizer),
        }
    )


def calculate_field_accuracy(auto: Dict[str, Any], reviewed: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the pipeline's automatic extraction (`auto`) against the
    same document's saved human review (`reviewed`) -- the ground truth.

    Persons are matched by `record_index` (their position in the original
    auto-extracted list) so a reviewer editing a record's name/ID doesn't
    break its association with the correct auto row. Persons the reviewer
    added have no `record_index` match (all-incorrect vs. blank auto side);
    persons the reviewer deleted show up as "Removed record" entries
    (all-incorrect vs. blank reviewed side) so hallucinated records still
    count against accuracy.

    Returns {"total_fields", "correct_fields", "incorrect_fields",
    "accuracy" (0-100), "field_results" (list)}.
    """
    auto_document = auto.get("document", {}) or {}
    reviewed_document = reviewed.get("document", {}) or {}
    auto_persons = auto.get("persons", []) or []
    reviewed_persons = reviewed.get("persons", []) or []

    field_results: List[Dict[str, Any]] = []

    document_field_defs = [
        ("court_name", "Court Name", _normalize_value),
        ("case_number", "Case Number", _normalize_value),
        ("document_number", "Document Number", _normalize_value),
        ("document_date", "Document Date", _normalize_date_value),
    ]
    for key, label, normalizer in document_field_defs:
        auto_value = _get_document_value(auto_document, key)
        reviewed_value = _get_document_value(reviewed_document, key)
        _add_field_result(field_results, label, auto_value, reviewed_value, normalizer)

    auto_by_index = {
        person.get("record_index"): person
        for person in auto_persons
        if person.get("record_index") is not None
    }
    consumed_indexes = set()

    for position, reviewed_person in enumerate(reviewed_persons, start=1):
        record_index = reviewed_person.get("record_index")
        auto_person = auto_by_index.get(record_index) if record_index is not None else None
        if auto_person is not None:
            consumed_indexes.add(record_index)
        auto_person = auto_person or {}

        person_type = reviewed_person.get("person_type") or auto_person.get("person_type")
        for key, normalizer in _person_field_defs(person_type):
            label = f"Person {position} - {_PERSON_FIELD_LABELS[key]}"
            _add_field_result(
                field_results, label, auto_person.get(key), reviewed_person.get(key), normalizer
            )

    for position, auto_person in enumerate(auto_persons, start=1):
        record_index = auto_person.get("record_index")
        if record_index is not None and record_index in consumed_indexes:
            continue

        person_type = auto_person.get("person_type")
        for key, normalizer in _person_field_defs(person_type):
            label = f"Removed record (was Person {position}) - {_PERSON_FIELD_LABELS[key]}"
            _add_field_result(field_results, label, auto_person.get(key), None, normalizer)

    total_fields = len(field_results)
    correct_fields = sum(1 for item in field_results if item["correct"])
    incorrect_fields = total_fields - correct_fields
    accuracy = round((correct_fields / total_fields) * 100, 1) if total_fields else 0.0

    return {
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "incorrect_fields": incorrect_fields,
        "accuracy": accuracy,
        "field_results": field_results,
    }
