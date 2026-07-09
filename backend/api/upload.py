"""POST /upload -- accept and store a court order file."""

from fastapi import APIRouter, HTTPException, UploadFile, File

from services.file_service import (
    is_allowed_file,
    is_allowed_size,
    generate_document_id,
    save_upload,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
)

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()

    if not is_allowed_size(len(content)):
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    document_id = generate_document_id()
    save_upload(document_id, file.filename, content)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "uploaded",
    }
