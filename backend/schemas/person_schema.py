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
    extraction_method: Optional[str] = None
    # Stable reference to this record's position in the original automatic
    # extraction, assigned once by api/process.py and preserved through the
    # review workflow (see streamlit_app/components/persons_table.py). Used
    # by evaluation_service.calculate_field_accuracy to associate a
    # human-reviewed row with the correct automatic record even after the
    # reviewer edits identifying fields like full_name or national_id --
    # never shown in the UI.
    record_index: Optional[int] = None