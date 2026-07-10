"""Crop and OCR the top-right header region of a page image.

Arabic Jordanian court letters put the reference number and date in a
small header block near the top-right corner. Full-page OCR often misses
or misreads it because it's small relative to the rest of the page.
Cropping just that region and running OCR on it separately -- then
placing that text first in the combined text -- gives header fields
(الرقم, التاريخ) a much better chance of being read correctly.

This never raises: any failure here just means the pipeline continues
with full-page OCR only, exactly as before this feature existed.
"""

from pathlib import Path
from typing import Any, Dict

from services.ocr_router_service import run_ocr_on_page

# Approximate header box for Jordanian court letters: top-right corner,
# expressed as fractions of the page width/height so it works regardless
# of scan resolution.
HEADER_X1_RATIO = 0.45
HEADER_X2_RATIO = 0.98
HEADER_Y1_RATIO = 0.05
HEADER_Y2_RATIO = 0.28

# The header crop is already small; upscale it further before OCR so
# small header text (case/document number, date) has enough pixels to
# be read reliably.
HEADER_UPSCALE_FACTOR = 3


def _crop_header_region(image_path: Path, output_path: Path) -> bool:
    """Crop the top-right header region of a page image, upscale it, and
    save it.

    Returns True on success, False if the crop could not be produced
    (missing cv2, unreadable image, degenerate crop, etc.).
    """
    try:
        import cv2
    except Exception:
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        return False

    height, width = image.shape[:2]
    x1 = int(width * HEADER_X1_RATIO)
    x2 = int(width * HEADER_X2_RATIO)
    y1 = int(height * HEADER_Y1_RATIO)
    y2 = int(height * HEADER_Y2_RATIO)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    crop = cv2.resize(
        crop, None,
        fx=HEADER_UPSCALE_FACTOR, fy=HEADER_UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    cv2.imwrite(str(output_path), crop)
    return True


def extract_header_text(page_number: int, image_path: Path) -> Dict[str, Any]:
    """Crop the header region of one page and run the OCR router on it.

    Never raises -- on any failure, returns a "failed" result with empty
    text so the caller can safely continue with full-page OCR only.
    """
    try:
        header_path = image_path.parent / f"{image_path.stem}_header{image_path.suffix}"

        if not _crop_header_region(image_path, header_path):
            return {
                "text": "",
                "status": "failed",
                "selected_engine": None,
                "average_confidence": 0.0,
                "quality_score": 0.0,
                "error": "Could not crop header region",
            }

        header_ocr = run_ocr_on_page(page_number, header_path)

        return {
            "text": header_ocr.get("text", "") or "",
            "status": "success" if header_ocr.get("selected_engine") else "failed",
            "selected_engine": header_ocr.get("selected_engine"),
            "average_confidence": header_ocr.get("average_confidence", 0.0),
            "quality_score": header_ocr.get("quality_score", 0.0),
            "error": None,
        }
    except Exception as exc:
        # Header OCR is a best-effort enhancement -- never let it break
        # the main pipeline.
        return {
            "text": "",
            "status": "failed",
            "selected_engine": None,
            "average_confidence": 0.0,
            "quality_score": 0.0,
            "error": f"Header OCR failed: {exc}",
        }
