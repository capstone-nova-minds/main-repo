"""GET /export/json/{document_id} and GET /export/excel/{document_id}.

Both use reviewed data when available, otherwise fall back to the raw
extracted data (see services/review_service.get_best_available_result).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.export_service import export_to_json, export_to_excel

router = APIRouter()


@router.get("/export/json/{document_id}")
def export_json(document_id: str):
    try:
        path = export_to_json(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return FileResponse(
        path=path,
        media_type="application/json",
        filename=f"{document_id}.json",
    )


@router.get("/export/excel/{document_id}")
def export_excel(document_id: str):
    try:
        path = export_to_excel(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return FileResponse(
        path=path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{document_id}.xlsx",
    )
