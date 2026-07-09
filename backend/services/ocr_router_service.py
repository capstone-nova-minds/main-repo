"""OCR Router / Fallback: tries EasyOCR first, falls back to Tesseract,
and picks whichever result has the higher quality score.

This is the only place other services should call into OCR -- they
never talk to a specific engine directly.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from services.ocr_engines.easyocr_engine import EasyOCREngine
from services.ocr_engines.tesseract_engine import TesseractEngine
from services.ocr_engines.paddleocr_engine_stub import PaddleOCREngineStub
from utils.ocr_quality import calculate_ocr_quality

QUALITY_THRESHOLD = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.65"))

# Engines are created once and reused (EasyOCR/Tesseract availability
# checks and model loads are expensive).
_easyocr_engine = EasyOCREngine()
_tesseract_engine = TesseractEngine()
_paddleocr_engine = PaddleOCREngineStub()


def _run_engine_attempt(engine, image_path: str) -> Dict[str, Any]:
    """Run one engine and wrap the result as an 'engine_attempts' entry."""
    try:
        if not engine.is_available():
            return {
                "engine": engine.engine_name,
                "status": "failed",
                "average_confidence": 0.0,
                "quality_score": 0.0,
                "text_length": 0,
                "error": f"{engine.engine_name} is not available in this environment",
                "text": "",
            }

        result = engine.extract_text(image_path)
        text = result.get("text", "") or ""
        confidence = result.get("average_confidence", 0.0) or 0.0
        quality = calculate_ocr_quality(text, confidence)

        return {
            "engine": engine.engine_name,
            "status": result.get("status", "failed"),
            "average_confidence": confidence,
            "quality_score": quality,
            "text_length": len(text),
            "error": result.get("error"),
            "text": text,
        }
    except Exception as exc:
        # An engine must never crash the pipeline.
        return {
            "engine": engine.engine_name,
            "status": "failed",
            "average_confidence": 0.0,
            "quality_score": 0.0,
            "text_length": 0,
            "error": f"Unexpected error: {exc}",
            "text": "",
        }


def run_ocr_on_page(page_number: int, image_path: Path) -> Dict[str, Any]:
    """Run the OCR router on a single page image and return the page result."""
    attempts: List[Dict[str, Any]] = []

    easyocr_attempt = _run_engine_attempt(_easyocr_engine, str(image_path))
    attempts.append(easyocr_attempt)

    best_attempt = easyocr_attempt
    needs_fallback = (
        easyocr_attempt["status"] != "success"
        or easyocr_attempt["quality_score"] < QUALITY_THRESHOLD
    )

    if needs_fallback:
        tesseract_attempt = _run_engine_attempt(_tesseract_engine, str(image_path))
        attempts.append(tesseract_attempt)

        if tesseract_attempt["quality_score"] > best_attempt["quality_score"]:
            best_attempt = tesseract_attempt
    else:
        attempts.append({
            "engine": _tesseract_engine.engine_name,
            "status": "skipped",
            "average_confidence": 0.0,
            "quality_score": 0.0,
            "text_length": 0,
            "error": None,
        })

    page_status = "success" if best_attempt["status"] == "success" else "failed"

    # Strip the raw "text" key out of attempt summaries (kept only on the
    # winning result at the page level) to match the documented OCR schema.
    attempt_summaries = [
        {k: v for k, v in a.items() if k != "text"} for a in attempts
    ]

    return {
        "page_number": page_number,
        "selected_engine": best_attempt["engine"] if page_status == "success" else None,
        "text": best_attempt.get("text", ""),
        "average_confidence": best_attempt["average_confidence"],
        "quality_score": best_attempt["quality_score"],
        "needs_review": page_status != "success" or best_attempt["quality_score"] < QUALITY_THRESHOLD,
        "engine_attempts": attempt_summaries,
    }


def run_ocr_router(document_id: str, page_paths: List[Path]) -> Dict[str, Any]:
    """Run OCR (with fallback) over every page of a document."""
    pages = [run_ocr_on_page(i + 1, path) for i, path in enumerate(page_paths)]

    successful_pages = [p for p in pages if p["selected_engine"]]
    ocr_status = "success" if successful_pages else "failed"

    # Document-level "selected_engine" reflects whichever engine won on
    # the most pages, defaulting to easyocr's name if nothing succeeded.
    engine_counts: Dict[str, int] = {}
    for p in successful_pages:
        engine_counts[p["selected_engine"]] = engine_counts.get(p["selected_engine"], 0) + 1
    selected_engine = max(engine_counts, key=engine_counts.get) if engine_counts else None

    return {
        "document_id": document_id,
        "selected_engine": selected_engine,
        "ocr_status": ocr_status,
        "pages": pages,
    }
