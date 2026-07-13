"""FastAPI entrypoint for the local AI Court Order Extraction backend.

Fully local: no LLM, no cloud AI, no cloud OCR. See docs/technical_documentation.md.
"""

from fastapi import FastAPI

from api import upload, process, review, export
from services.file_service import ensure_data_dirs

app = FastAPI(title="Court Order Extraction API")

ensure_data_dirs()

app.include_router(upload.router)
app.include_router(process.router)
app.include_router(review.router)
app.include_router(export.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "court-order-backend"}
