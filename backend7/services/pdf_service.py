"""Convert uploaded documents into per-page PNG images using PyMuPDF.

If the upload is already an image (jpg/jpeg/png), it is treated as a
single page 1 and copied into the pages folder unchanged.
"""

import shutil
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from services.file_service import PAGES_DIR

RENDER_DPI = 200


def _page_dir(document_id: str) -> Path:
    folder = PAGES_DIR / document_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def convert_pdf_to_images(document_id: str, pdf_path: Path) -> List[Path]:
    """Render every PDF page to a PNG file. Returns the list of image paths."""
    folder = _page_dir(document_id)
    zoom = RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    page_paths: List[Path] = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            out_path = folder / f"page_{page_index + 1}.png"
            pix.save(out_path)
            page_paths.append(out_path)

    return page_paths


def copy_image_as_page_one(document_id: str, image_path: Path) -> List[Path]:
    """Treat a directly-uploaded image as page 1."""
    folder = _page_dir(document_id)
    out_path = folder / "page_1.png"
    shutil.copyfile(image_path, out_path)
    return [out_path]


def get_document_pages(document_id: str, upload_path: Path, extension: str) -> List[Path]:
    """Entry point used by the processing pipeline: PDF -> pages, image -> page 1."""
    if extension == ".pdf":
        return convert_pdf_to_images(document_id, upload_path)
    return copy_image_as_page_one(document_id, upload_path)
