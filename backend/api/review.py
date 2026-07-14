"""POST /review/{document_id} -- save human-edited extraction results."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from services.evaluation_service import calculate_field_accuracy
from services.review_service import load_extracted_result, save_reviewed_result

router = APIRouter()


@router.post("/review/{document_id}")
def save_review(document_id: str, reviewed_data: Dict[str, Any]):
    try:
        save_reviewed_result(document_id, reviewed_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save reviewed result: {exc}")

    # Measured Field Accuracy: compare the auto-extracted result (before
    # this review) against what the reviewer just saved (ground truth).
    # Only possible if the raw auto-extracted result is still on disk.
    auto_result = load_extracted_result(document_id)
    evaluation = calculate_field_accuracy(auto_result, reviewed_data) if auto_result else None

    return {
        "document_id": document_id,
        "status": "reviewed_saved",
        "evaluation": evaluation,
    }
