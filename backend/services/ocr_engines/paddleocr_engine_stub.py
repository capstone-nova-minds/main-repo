"""Placeholder for a future PaddleOCR Arabic engine.

PaddleOCR is not implemented yet. This stub exists so the OCR router
can be extended later without changing its interface -- just flip
ENABLE_PADDLEOCR and implement extract_text() using PaddleOCR's API.
"""

from typing import Any, Dict

from services.ocr_engines.base_ocr_engine import BaseOCREngine


class PaddleOCREngineStub(BaseOCREngine):
    engine_name = "paddleocr"

    def is_available(self) -> bool:
        # Always False until PaddleOCR Arabic support is actually added.
        return False

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        return {
            "text": "",
            "average_confidence": 0.0,
            "status": "failed",
            "error": "PaddleOCR engine is not implemented yet",
        }
