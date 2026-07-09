"""File handling: saving uploads, validating type/size, path helpers.

All paths are relative to backend/data (mounted as a Docker volume so
files persist and are shared with the host / other future consumers).
"""

import uuid
from pathlib import Path
from typing import Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PAGES_DIR = DATA_DIR / "pages"
PROCESSED_DIR = DATA_DIR / "processed"
OCR_OUTPUTS_DIR = DATA_DIR / "ocr_outputs"
EXTRACTED_DIR = DATA_DIR / "extracted"
REVIEWED_DIR = DATA_DIR / "reviewed"
EXPORTS_DIR = DATA_DIR / "exports"

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE_MB = 20


def ensure_data_dirs() -> None:
    """Create all data subfolders if they don't exist (safe to call anytime)."""
    for folder in [
        UPLOADS_DIR, PAGES_DIR, PROCESSED_DIR,
        OCR_OUTPUTS_DIR, EXTRACTED_DIR, REVIEWED_DIR, EXPORTS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_allowed_size(size_bytes: int) -> bool:
    return size_bytes <= MAX_UPLOAD_SIZE_MB * 1024 * 1024


def generate_document_id() -> str:
    return str(uuid.uuid4())


def save_upload(document_id: str, filename: str, content: bytes) -> Path:
    """Save uploaded file bytes under data/uploads/{document_id}{ext}."""
    ensure_data_dirs()
    ext = Path(filename).suffix.lower()
    dest = UPLOADS_DIR / f"{document_id}{ext}"
    dest.write_bytes(content)
    return dest


def find_uploaded_file(document_id: str) -> Tuple[Path, str]:
    """Find the uploaded file for a document_id regardless of extension."""
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOADS_DIR / f"{document_id}{ext}"
        if candidate.exists():
            return candidate, ext
    raise FileNotFoundError(f"No uploaded file found for document_id={document_id}")
