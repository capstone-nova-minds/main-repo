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

from typing import Any, Dict, Optional

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
