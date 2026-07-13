"""Save and load human-reviewed extraction results.

Reviewed data always takes precedence over raw extracted data once it
exists -- export_service relies on that ordering.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from services.file_service import EXTRACTED_DIR, REVIEWED_DIR, ensure_data_dirs
from services.evaluation_service import calculate_field_accuracy
from utils.json_utils import make_json_serializable

logger = logging.getLogger(__name__)


def _extracted_path(document_id: str) -> Path:
    return EXTRACTED_DIR / f"{document_id}.json"


def _reviewed_path(document_id: str) -> Path:
    return REVIEWED_DIR / f"{document_id}.json"


def save_extracted_result(document_id: str, result: Dict[str, Any]) -> None:
    ensure_data_dirs()
    safe_result = make_json_serializable(result)
    _extracted_path(document_id).write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_extracted_result(document_id: str) -> Optional[Dict[str, Any]]:
    path = _extracted_path(document_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_reviewed_result(document_id: str, reviewed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a human review and measure its field accuracy against the
    original automatic extraction (never against the reviewed data itself,
    and never re-derived from OCR/NER confidence).

    The original automatic extraction is only ever *read* here
    (load_extracted_result), never overwritten -- it lives in a separate
    file/directory (EXTRACTED_DIR) from the reviewed result (REVIEWED_DIR),
    so re-saving a review can never corrupt the ground truth it's compared
    against.

    Returns the exact dict that was persisted, including the computed
    "evaluation" (or None if there is no original automatic extraction to
    compare against -- handled safely rather than raising).
    """
    ensure_data_dirs()
    reviewed_data["reviewed"] = True

    auto_result = load_extracted_result(document_id)

    if auto_result is not None:
        reviewed_data["evaluation"] = calculate_field_accuracy(auto_result, reviewed_data)
    else:
        logger.warning(
            "document_id=%s no_original_extraction_found_evaluation_skipped",
            document_id,
        )
        reviewed_data["evaluation"] = None

    safe_data = make_json_serializable(reviewed_data)
    _reviewed_path(document_id).write_text(
        json.dumps(safe_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return safe_data


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
