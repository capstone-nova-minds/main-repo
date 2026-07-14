"""Editable form for document-level extracted fields."""

import streamlit as st

DOCUMENT_FIELDS = ["court_name", "case_number", "document_number", "document_date"]
FIELD_LABELS = {
    "court_name": "Court Name",
    "case_number": "Case Number",
    "document_number": "Document Number",
    "document_date": "Document Date",
}


def render_document_fields_form(document: dict, document_id: str = "") -> dict:
    """Render editable inputs for each document field, return the edited dict.

    document_id is mixed into each widget's key so Streamlit treats the
    fields as brand-new widgets when the reviewer switches documents --
    otherwise it keeps showing the previously selected document's cached
    values instead of the new one's.
    """
    edited = {}

    for field_name in DOCUMENT_FIELDS:
        field = document.get(field_name, {}) or {}
        current_value = field.get("value") or ""
        needs_review = field.get("needs_review", False)

        label = FIELD_LABELS[field_name]
        if needs_review:
            label += " ⚠️ needs review"

        new_value = st.text_input(label, value=current_value, key=f"doc_field_{document_id}_{field_name}")

        edited[field_name] = {
            "value": new_value if new_value else None,
            "confidence": field.get("confidence", 0.0),
            "needs_review": needs_review and new_value == current_value,
        }

    return edited
