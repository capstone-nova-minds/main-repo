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
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.arabic_normalizer import normalize_arabic_digits
from services.person_extraction_service import (
    _normalize_registration_number as _normalize_registration_number_raw,
)

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
# Multi-person evaluation
# ---------------------------------------------------------------------------
# evaluate_accuracy() above only ever looks at the first person, by design
# (see module docstring). Structured target-list documents routinely list
# several persons/companies on one page, so this variant matches expected
# and actual persons by identifier (national_id or registration_number)
# and scores every person-level field for every expected person, alongside
# the same document-level fields. This is additive -- it does not change
# evaluate_accuracy()'s behavior or shape.

MULTI_PERSON_FIELDS = ["full_name", "national_id", "registration_number", "person_type"]


def _person_identifier(person: Dict[str, Any]) -> Optional[str]:
    national_id = person.get("national_id")
    if national_id:
        return f"id:{national_id}"

    registration_number = person.get("registration_number")
    if registration_number:
        return f"reg:{registration_number}"

    return None


def _match_actual_person(
    expected_person: Dict[str, Any],
    actual_persons: list,
) -> Dict[str, Any]:
    """Find the actual person with the same identifier as expected_person.

    Falls back to positional matching only when neither record carries an
    identifier (e.g. a name-only expected fixture).
    """
    expected_key = _person_identifier(expected_person)

    if expected_key:
        for actual_person in actual_persons:
            if _person_identifier(actual_person) == expected_key:
                return actual_person
        return {}

    expected_name = (expected_person.get("full_name") or "").strip()

    for actual_person in actual_persons:
        if (actual_person.get("full_name") or "").strip() == expected_name:
            return actual_person

    return {}


def evaluate_accuracy_multi(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Field-level accuracy across *every* expected person/company, not
    just the first one.

    Returns {"total_fields", "correct_fields", "accuracy", "field_results"}
    where field_results keys are "document.<field>" and
    "person[<index>].<field>" so a caller can see exactly which field on
    which person/company record was wrong.
    """
    expected_document = expected.get("document", {})
    actual_document = actual.get("document", {})

    expected_persons = expected.get("persons") or []
    actual_persons = actual.get("persons") or []

    field_results: Dict[str, bool] = {}

    for field_name in DOCUMENT_FIELDS:
        expected_value = _get_document_value(expected_document, field_name)
        actual_value = _get_document_value(actual_document, field_name)
        field_results[f"document.{field_name}"] = _values_match(expected_value, actual_value)

    for index, expected_person in enumerate(expected_persons):
        matched_actual = _match_actual_person(expected_person, actual_persons)

        for field_name in MULTI_PERSON_FIELDS:
            expected_value = expected_person.get(field_name)
            actual_value = matched_actual.get(field_name)
            field_results[f"person[{index}].{field_name}"] = _values_match(
                expected_value, actual_value
            )

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
# Measured field accuracy (automatic extraction vs. human-reviewed values)
# ---------------------------------------------------------------------------
# This is a *different* metric from evaluate_accuracy/evaluate_accuracy_multi
# above: those compare a result against a hand-labeled test fixture, for
# tracking this codebase's own extraction quality during development.
# calculate_field_accuracy below is the production-facing metric: it
# compares the automatic extraction against what a human reviewer actually
# approved for one real document, treating the reviewed values as ground
# truth. It only ever runs after a human review has been saved, and never
# looks at OCR/NER confidence -- see backend/services/review_service.py.

_DIACRITICS_PATTERN = re.compile(
    "[" + "ؐ-ؚ" + "ً-ٟ" + "ٰ" + "ۖ-ۜ"
    + "۟-ۨ" + "۪-ۭ" + "]"
)
_TATWEEL = "ـ"
_ALEF_VARIANTS_TRANS = str.maketrans({
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "آ": "ا",  # آ -> ا
    "ٱ": "ا",  # ٱ -> ا
})
_ALEF_MAKSURA = "ى"  # ى
_YA = "ي"  # ي
_HARMLESS_PUNCTUATION_PATTERN = re.compile(r"[.,;:،؛!?'\"()\[\]{}|]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def _normalize_arabic_value(value: Optional[str]) -> Optional[str]:
    """Deterministic Arabic-text normalization for accuracy comparison.

    Trims and collapses whitespace, strips diacritics (tashkeel) and
    tatweel, folds hamza-alef variants (alef with hamza above/below/madda,
    and alef wasla) to a plain alef and alef-maksura to ya, and ignores
    harmless punctuation. Never reorders words and never applies
    fuzzy/approximate matching -- the final comparison is always an exact
    string match on the normalized form.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = _DIACRITICS_PATTERN.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_VARIANTS_TRANS)
    text = text.replace(_ALEF_MAKSURA, _YA)
    text = _HARMLESS_PUNCTUATION_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text or None


def _normalize_national_id_value(value: Optional[str]) -> Optional[str]:
    """ASCII-digit-only normalization for National ID comparison.

    Converts Arabic-Indic/Persian digits to ASCII and strips everything
    that isn't a digit (spaces, punctuation). Does not pad or truncate --
    a value that isn't 11 digits after normalization is still compared as
    whatever digits it has, so a genuinely different/wrong ID is still
    caught as incorrect rather than silently coerced.
    """
    if value is None:
        return None

    text = normalize_arabic_digits(str(value))
    text = re.sub(r"\D", "", text)

    return text or None


def _normalize_registration_number_value(value: Optional[str]) -> Optional[str]:
    """Normalize a company registration number (REG-202701, REG - 202701,
    REG 202701, or a purely numeric registry number) to its canonical
    "REG-202701" form, reusing the same normalizer person_extraction_service
    uses so both stay consistent.

    Falls back to a whitespace/case-normalized form (rather than None) for
    a value that doesn't match the expected shape, so two different
    malformed values are never treated as equal just because neither
    parsed.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = _normalize_registration_number_raw(text)
    if normalized:
        return normalized

    return re.sub(r"\s+", "", text).upper()


_DOCUMENT_NUMBER_PATTERN = re.compile(r"^([A-Z]+)-?(\d{4})-?(\d+)$")


def _normalize_document_number_value(value: Optional[str]) -> Optional[str]:
    """Normalize harmless spacing/hyphen differences in a document/book
    number -- "U W-2026-0101" and "UW - 2026 - 0101" both normalize to
    "UW-2026-0101" -- but never invents a missing letter: "U-2026-0101"
    stays a genuinely different value from "UW-2026-0101", so a corrected
    "W" is still counted as a real (correct) fix.
    """
    if value is None:
        return None

    text = re.sub(r"\s+", "", str(value)).upper()
    text = re.sub(r"[-–—]+", "-", text)

    if not text:
        return None

    match = _DOCUMENT_NUMBER_PATTERN.match(text)
    if match:
        letters, year, serial = match.groups()
        return f"{letters}-{year}-{serial}"

    return text


_DATE_PATTERN = re.compile(r"^(\d{1,4})/(\d{1,2})/(\d{1,4})$")


def _normalize_date_value(value: Optional[str]) -> Optional[str]:
    """Normalize DD/MM/YYYY, YYYY/MM/DD, and hyphen-separated variants of
    a date to YYYY/MM/DD."""
    if value is None:
        return None

    text = str(value).strip().replace("-", "/")
    if not text:
        return None

    match = _DATE_PATTERN.match(text)
    if not match:
        return text

    first, second, third = match.groups()

    if len(first) == 4:
        year, month, day = first, second, third
    elif len(third) == 4:
        day, month, year = first, second, third
    else:
        return text

    try:
        return f"{year}/{int(month):02d}/{int(day):02d}"
    except ValueError:
        return text


def _normalize_case_number_value(value: Optional[str]) -> Optional[str]:
    """Normalize harmless whitespace around the case number's '/' separator."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"\s*/\s*", "/", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text or None


def _normalize_plain_value(value: Optional[str]) -> Optional[str]:
    """Trim-only normalization for fields with no special shape (e.g.
    person_type)."""
    if value is None:
        return None

    text = str(value).strip()
    return text or None


_FIELD_NORMALIZERS = {
    "court_name": _normalize_arabic_value,
    "case_number": _normalize_case_number_value,
    "document_number": _normalize_document_number_value,
    "document_date": _normalize_date_value,
    "full_name": _normalize_arabic_value,
    "national_id": _normalize_national_id_value,
    "registration_number": _normalize_registration_number_value,
    "person_type": _normalize_plain_value,
}


def _values_equal(field_key: str, auto_value: Optional[str], reviewed_value: Optional[str]) -> bool:
    """Deterministic equality check for one field, using the normalizer
    for its shape. Missing-vs-missing (both None after normalization)
    counts as correct; missing-vs-present or present-vs-different does not.
    """
    normalizer = _FIELD_NORMALIZERS.get(field_key, _normalize_plain_value)
    return normalizer(auto_value) == normalizer(reviewed_value)


DOCUMENT_FIELD_LABELS: List[Tuple[str, str]] = [
    ("court_name", "Court Name"),
    ("case_number", "Case Number"),
    ("document_number", "Document Number"),
    ("document_date", "Document Date"),
]

INDIVIDUAL_FIELD_LABELS: List[Tuple[str, str]] = [
    ("full_name", "Full Name"),
    ("national_id", "National ID"),
    ("person_type", "Person Type"),
]

COMPANY_FIELD_LABELS: List[Tuple[str, str]] = [
    ("full_name", "Full Name"),
    ("registration_number", "Registration Number"),
    ("person_type", "Person Type"),
]


def _fields_for_person_type(person_type: Optional[str]) -> List[Tuple[str, str]]:
    """Company records are evaluated on registration_number, never
    national_id; Individual records the reverse -- never both."""
    return COMPANY_FIELD_LABELS if person_type == "Company" else INDIVIDUAL_FIELD_LABELS


def _build_field_result(
    label: str,
    field_key: str,
    auto_value: Optional[str],
    reviewed_value: Optional[str],
) -> Dict[str, Any]:
    return {
        "field": label,
        "auto_value": auto_value,
        "reviewed_value": reviewed_value,
        "correct": _values_equal(field_key, auto_value, reviewed_value),
    }


def _get_record_index(person: Dict[str, Any]) -> Optional[int]:
    """Read the stable record_index api/process.py stamps on each
    automatic record and the review UI preserves as a hidden column."""
    raw = person.get("record_index")
    if raw is None:
        return None

    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _match_auto_person(
    reviewed_person: Dict[str, Any],
    auto_persons: List[Dict[str, Any]],
    position: int,
    same_count: bool,
    matched_auto_indices: Set[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """Find the automatic record a reviewed record corresponds to.

    Priority:
    1. The stable record_index stamped by api/process.py and preserved
       through the review UI -- correct even if the reviewer edited every
       identifying field on the row.
    2. National ID (for whichever record actually has one).
    3. Registration number (for whichever record actually has one).
    4. The same positional index, but only when the reviewed list is the
       same length as the automatic list (no rows added/removed) -- once
       counts differ, position no longer reliably means the same record.

    Never matches by full_name: that is frequently the exact field being
    corrected, so using it to *find* the record would beg the question.

    Returns (matched_auto_person_or_none, matched_auto_index_or_none).
    """
    record_index = _get_record_index(reviewed_person)
    if (
        record_index is not None
        and 0 <= record_index < len(auto_persons)
        and record_index not in matched_auto_indices
    ):
        return auto_persons[record_index], record_index

    reviewed_national_id = _normalize_national_id_value(reviewed_person.get("national_id"))
    if reviewed_national_id:
        for idx, auto_person in enumerate(auto_persons):
            if idx in matched_auto_indices:
                continue
            if _normalize_national_id_value(auto_person.get("national_id")) == reviewed_national_id:
                return auto_person, idx

    reviewed_registration = _normalize_registration_number_value(
        reviewed_person.get("registration_number")
    )
    if reviewed_registration:
        for idx, auto_person in enumerate(auto_persons):
            if idx in matched_auto_indices:
                continue
            if (
                _normalize_registration_number_value(auto_person.get("registration_number"))
                == reviewed_registration
            ):
                return auto_person, idx

    if same_count and position < len(auto_persons) and position not in matched_auto_indices:
        return auto_persons[position], position

    return None, None


def calculate_field_accuracy(
    auto_result: Dict[str, Any],
    reviewed_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Measured field accuracy: a deterministic, field-by-field comparison
    of the original automatic extraction against the final human-reviewed
    values, which are treated as ground truth.

    Only ever call this after a human review has been saved -- it must
    never be computed from OCR/NER confidence, and never before a review
    exists (see services/review_service.save_reviewed_result).

    Evaluated fields:
    - Document: Court Name, Case Number, Document Number, Document Date.
    - Every Individual: Full Name, National ID, Person Type.
    - Every Company: Full Name, Registration Number, Person Type.

    A reviewed person with no corresponding automatic record (added during
    review) is scored against an all-missing automatic record, so every
    one of its evaluated fields is incorrect. An automatic record with no
    corresponding reviewed person (removed as a false positive) likewise
    contributes every one of its evaluated fields as incorrect.

    Returns:
    {
        "accuracy": 84.6,
        "correct_fields": 11,
        "incorrect_fields": 2,
        "total_fields": 13,
        "field_results": [
            {"field": "Court Name", "auto_value": ..., "reviewed_value": ..., "correct": true},
            ...
        ],
    }
    """
    auto_result = auto_result or {}
    reviewed_result = reviewed_result or {}

    field_results: List[Dict[str, Any]] = []

    auto_document = auto_result.get("document") or {}
    reviewed_document = reviewed_result.get("document") or {}

    for field_key, label in DOCUMENT_FIELD_LABELS:
        auto_value = _get_document_value(auto_document, field_key)
        reviewed_value = _get_document_value(reviewed_document, field_key)
        field_results.append(_build_field_result(label, field_key, auto_value, reviewed_value))

    auto_persons = auto_result.get("persons") or []
    reviewed_persons = reviewed_result.get("persons") or []
    same_count = len(auto_persons) == len(reviewed_persons)

    matched_auto_indices: Set[int] = set()

    for position, reviewed_person in enumerate(reviewed_persons):
        auto_person, auto_index = _match_auto_person(
            reviewed_person, auto_persons, position, same_count, matched_auto_indices
        )

        if auto_index is not None:
            matched_auto_indices.add(auto_index)

        person_type = (
            reviewed_person.get("person_type")
            or (auto_person or {}).get("person_type")
            or "Individual"
        )

        for field_key, label in _fields_for_person_type(person_type):
            auto_value = (auto_person or {}).get(field_key)
            reviewed_value = reviewed_person.get(field_key)
            field_results.append(
                _build_field_result(
                    f"Person {position + 1} - {label}", field_key, auto_value, reviewed_value
                )
            )

    # Automatic records the reviewer removed as false positives: there is
    # no reviewed value to have matched, so every evaluated field on them
    # counts as incorrect.
    for auto_index, auto_person in enumerate(auto_persons):
        if auto_index in matched_auto_indices:
            continue

        person_type = auto_person.get("person_type") or "Individual"

        for field_key, label in _fields_for_person_type(person_type):
            field_results.append(
                {
                    "field": f"Removed record (was Person {auto_index + 1}) - {label}",
                    "auto_value": auto_person.get(field_key),
                    "reviewed_value": None,
                    "correct": False,
                }
            )

    total_fields = len(field_results)
    correct_fields = sum(1 for item in field_results if item["correct"])
    incorrect_fields = total_fields - correct_fields
    accuracy = round((correct_fields / total_fields) * 100, 1) if total_fields else 0.0

    return {
        "accuracy": accuracy,
        "correct_fields": correct_fields,
        "incorrect_fields": incorrect_fields,
        "total_fields": total_fields,
        "field_results": field_results,
    }
