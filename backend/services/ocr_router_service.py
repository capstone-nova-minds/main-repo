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

QUALITY_THRESHOLD = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.75"))

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
        confidence = float(result.get("average_confidence", 0.0) or 0.0)
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


def run_ocr_on_page(
    page_number: int,
    image_path: Path,
) -> Dict[str, Any]:
    """Run PaddleOCR first, then EasyOCR, then Tesseract."""

    attempts: List[Dict[str, Any]] = []

    engines = [
        _paddleocr_engine,
        _easyocr_engine,
        _tesseract_engine,
    ]

    best_attempt: Dict[str, Any] | None = None
    selected_index: int | None = None

    for index, engine in enumerate(engines):
        attempt = _run_engine_attempt(
            engine,
            str(image_path),
        )

        attempts.append(attempt)

        if best_attempt is None:
            best_attempt = attempt
        else:
            best_success = best_attempt["status"] == "success"
            current_success = attempt["status"] == "success"

            if current_success and not best_success:
                best_attempt = attempt

            elif current_success == best_success:
                if (
                    attempt["quality_score"]
                    > best_attempt["quality_score"]
                ):
                    best_attempt = attempt

                elif (
                    attempt["quality_score"]
                    == best_attempt["quality_score"]
                    and attempt["average_confidence"]
                    > best_attempt["average_confidence"]
                ):
                    best_attempt = attempt

        if (
            attempt["status"] == "success"
            and attempt["quality_score"] >= QUALITY_THRESHOLD
        ):
            selected_index = index
            break

    # Add skipped summaries for engines that were not needed.
    if selected_index is not None:
        for engine in engines[selected_index + 1:]:
            attempts.append({
                "engine": engine.engine_name,
                "status": "skipped",
                "average_confidence": 0.0,
                "quality_score": 0.0,
                "text_length": 0,
                "error": None,
                "text": "",
            })

    if best_attempt is None:
        best_attempt = {
            "engine": None,
            "status": "failed",
            "average_confidence": 0.0,
            "quality_score": 0.0,
            "text_length": 0,
            "error": "No OCR engine was executed",
            "text": "",
        }

    page_status = (
        "success"
        if best_attempt["status"] == "success"
        and bool(best_attempt.get("text", "").strip())
        else "failed"
    )

    attempt_summaries = [
        {
            key: value
            for key, value in attempt.items()
            if key != "text"
        }
        for attempt in attempts
    ]

    return {
        "page_number": page_number,
        "selected_engine": (
            best_attempt["engine"]
            if page_status == "success"
            else None
        ),
        "text": best_attempt.get("text", ""),
        "average_confidence": best_attempt[
            "average_confidence"
        ],
        "quality_score": best_attempt["quality_score"],
        "needs_review": bool(
            page_status != "success"
            or best_attempt["quality_score"]
            < QUALITY_THRESHOLD
        ),
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
