"""Save and load human-reviewed extraction results.

Reviewed data always takes precedence over raw extracted data once it
exists -- export_service relies on that ordering.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from services.file_service import EXTRACTED_DIR, REVIEWED_DIR, ensure_data_dirs


def _extracted_path(document_id: str) -> Path:
    return EXTRACTED_DIR / f"{document_id}.json"


def _reviewed_path(document_id: str) -> Path:
    return REVIEWED_DIR / f"{document_id}.json"


def save_extracted_result(document_id: str, result: Dict[str, Any]) -> None:
    ensure_data_dirs()
    _extracted_path(document_id).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_extracted_result(document_id: str) -> Optional[Dict[str, Any]]:
    path = _extracted_path(document_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_reviewed_result(document_id: str, reviewed_data: Dict[str, Any]) -> None:
    ensure_data_dirs()
    reviewed_data["reviewed"] = True
    _reviewed_path(document_id).write_text(
        json.dumps(reviewed_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_reviewed_result(document_id: str) -> Optional[Dict[str, Any]]:
    path = _reviewed_path(document_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_best_available_result(document_id: str) -> Optional[Dict[str, Any]]:
    """Reviewed data wins if present, otherwise fall back to raw extracted data."""
    reviewed = load_reviewed_result(document_id)
    if reviewed is not None:
        return reviewed
    extracted = load_extracted_result(document_id)
    if extracted is not None:
        extracted.setdefault("reviewed", False)
    return extracted
