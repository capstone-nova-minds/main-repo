"""Validate document and person fields, finalizing confidence/needs_review flags."""

import re
from typing import List

from schemas.document_schema import DocumentInfo
from schemas.person_schema import PersonRecord
from utils.confidence import CONFIDENCE_THRESHOLD

_NATIONAL_ID_PATTERN = re.compile(r"^\d{11}$")
_VALID_PERSON_TYPES = {"Individual", "Company"}


def is_valid_national_id(national_id) -> bool:
    if national_id is None:
        return True  # missing is allowed -- just means null
    return bool(_NATIONAL_ID_PATTERN.match(str(national_id)))


def validate_document(document: DocumentInfo) -> DocumentInfo:
    """Ensure missing mandatory fields are flagged for review."""
    for field_name in ["court_name", "case_number", "document_number", "document_date"]:
        field = getattr(document, field_name)
        if not field.value:
            field.needs_review = True
            field.confidence = 0.0
        elif field.confidence < CONFIDENCE_THRESHOLD:
            field.needs_review = True
    return document


def validate_person(person: PersonRecord) -> PersonRecord:
    """Apply business rules to a single person/company record."""
    if not person.full_name:
        person.needs_review = True

    if person.person_type not in _VALID_PERSON_TYPES:
        person.person_type = "Individual"
        person.needs_review = True

    if not is_valid_national_id(person.national_id):
        # Invalid IDs are dropped rather than trusted, per "do not invent
        # missing values" -- an 11-digit mismatch is likely an OCR error.
        person.national_id = None
        person.needs_review = True

    if person.person_type == "Company" and person.national_id:
        # Companies use registration_number, not national_id.
        person.national_id = None
        person.needs_review = True

    if person.confidence < CONFIDENCE_THRESHOLD:
        person.needs_review = True

    return person


def validate_all(document: DocumentInfo, persons: List[PersonRecord]):
    validated_document = validate_document(document)
    validated_persons = [validate_person(p) for p in persons]
    return validated_document, validated_persons
