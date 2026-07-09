"""Pydantic model for a single person/company record."""

from typing import Optional

from pydantic import BaseModel


class PersonRecord(BaseModel):
    """A person or company listed in the court order."""

    full_name: Optional[str] = None
    national_id: Optional[str] = None
    registration_number: Optional[str] = None
    person_type: str = "Individual"  # "Individual" or "Company"
    confidence: float = 0.0
    needs_review: bool = True
    source: str = "rules"  # "rules", "ner", or "rules+ner"
