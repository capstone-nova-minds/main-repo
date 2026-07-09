"""Export approved (reviewed, or extracted-as-fallback) results to JSON/Excel.

Excel: one row per person, flattened with document-level fields repeated
on every row so the sheet is self-contained.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from services.file_service import EXPORTS_DIR, ensure_data_dirs
from services.review_service import get_best_available_result

EXCEL_COLUMNS = [
    "document_id",
    "court_name",
    "case_number",
    "document_number",
    "document_date",
    "full_name",
    "national_id",
    "registration_number",
    "person_type",
    "confidence",
    "needs_review",
]


def _field_value(document: Dict[str, Any], field_name: str):
    field = document.get(field_name) or {}
    return field.get("value")


def export_to_json(document_id: str) -> Path:
    result = get_best_available_result(document_id)
    if result is None:
        raise FileNotFoundError(f"No extracted or reviewed result for document_id={document_id}")

    ensure_data_dirs()
    out_path = EXPORTS_DIR / f"{document_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_to_excel(document_id: str) -> Path:
    result = get_best_available_result(document_id)
    if result is None:
        raise FileNotFoundError(f"No extracted or reviewed result for document_id={document_id}")

    document = result.get("document", {})
    persons = result.get("persons", []) or [{}]  # at least one row, even with no persons

    rows = []
    for person in persons:
        rows.append({
            "document_id": document_id,
            "court_name": _field_value(document, "court_name"),
            "case_number": _field_value(document, "case_number"),
            "document_number": _field_value(document, "document_number"),
            "document_date": _field_value(document, "document_date"),
            "full_name": person.get("full_name"),
            "national_id": person.get("national_id"),
            "registration_number": person.get("registration_number"),
            "person_type": person.get("person_type"),
            "confidence": person.get("confidence"),
            "needs_review": person.get("needs_review"),
        })

    df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)

    ensure_data_dirs()
    out_path = EXPORTS_DIR / f"{document_id}.xlsx"
    df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path
