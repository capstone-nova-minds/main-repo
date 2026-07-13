"""POST /review/{document_id} -- save human-edited extraction results."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from services.review_service import save_reviewed_result

router = APIRouter()


@router.post("/review/{document_id}")
def save_review(document_id: str, reviewed_data: Dict[str, Any]):
    try:
        saved = save_reviewed_result(document_id, reviewed_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save reviewed result: {exc}")

    return {
        "document_id": document_id,
        "status": "reviewed_saved",
        "evaluation": saved.get("evaluation"),
    }
