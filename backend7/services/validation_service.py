"""Validate document and person fields, finalizing confidence/needs_review flags."""

import re
from typing import List

from schemas.document_schema import DocumentInfo
from schemas.person_schema import PersonRecord
from utils.confidence import CONFIDENCE_THRESHOLD

_NATIONAL_ID_PATTERN = re.compile(r"^\d{11}$")
_NATIONAL_ID_LOOSE_PATTERN = re.compile(r"^\d{10,11}$")
_VALID_PERSON_TYPES = {"Individual", "Company"}

_CASE_NUMBER_SHAPE = re.compile(r"^(\d{1,6}/(19|20)\d{2}|(19|20)\d{2}/\d{1,6})$")
_DOCUMENT_NUMBER_SHAPE = re.compile(r"^[A-Z]{1,10}-(19|20)\d{2}-\d{2,10}$")
_DOCUMENT_DATE_SHAPE = re.compile(r"^(19|20)\d{2}/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])$")


def is_valid_national_id(national_id) -> bool:
    if national_id is None:
        return True
    return bool(_NATIONAL_ID_PATTERN.match(str(national_id)))


def is_plausible_national_id(national_id) -> bool:
    if national_id is None:
        return False
    return bool(_NATIONAL_ID_LOOSE_PATTERN.match(str(national_id)))


_FIELD_SHAPE_CHECKS = {
    "case_number": _CASE_NUMBER_SHAPE,
    "document_number": _DOCUMENT_NUMBER_SHAPE,
    "document_date": _DOCUMENT_DATE_SHAPE,
}


def validate_document(document: DocumentInfo) -> DocumentInfo:
    for field_name in ["court_name", "case_number", "document_number", "document_date"]:
        field = getattr(document, field_name)

        shape_check = _FIELD_SHAPE_CHECKS.get(field_name)

        if field.value and shape_check and not shape_check.match(str(field.value)):
            field.value = None
            field.confidence = 0.0
            field.needs_review = True
            continue

        if not field.value:
            field.needs_review = True
            field.confidence = 0.0
        elif field.confidence < CONFIDENCE_THRESHOLD:
            field.needs_review = True
    return document


def validate_person(person: PersonRecord) -> PersonRecord:
    if not person.full_name:
        person.needs_review = True

    if person.person_type not in _VALID_PERSON_TYPES:
        person.person_type = "Individual"
        person.needs_review = True

    if person.national_id and not is_valid_national_id(person.national_id):
        if is_plausible_national_id(person.national_id):
            person.needs_review = True
        else:
            person.national_id = None
            person.needs_review = True

    if person.person_type == "Company" and person.national_id:
        person.national_id = None
        person.needs_review = True

    if person.confidence < CONFIDENCE_THRESHOLD:
        person.needs_review = True

    return person


def validate_all(document: DocumentInfo, persons: List[PersonRecord]):
    validated_document = validate_document(document)
    validated_persons = [validate_person(p) for p in persons]
    return validated_document, validated_persons