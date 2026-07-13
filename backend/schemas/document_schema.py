"""Pydantic models for document-level extracted fields."""

from typing import Optional

from pydantic import BaseModel


class ExtractedField(BaseModel):
    """A single extracted value with its confidence and review flag."""

    value: Optional[str] = None
    confidence: float = 0.0
    needs_review: bool = True


class DocumentInfo(BaseModel):
    """Document-level fields extracted from a court attachment order."""

    court_name: ExtractedField = ExtractedField()
    case_number: ExtractedField = ExtractedField()
    document_number: ExtractedField = ExtractedField()
    document_date: ExtractedField = ExtractedField()
