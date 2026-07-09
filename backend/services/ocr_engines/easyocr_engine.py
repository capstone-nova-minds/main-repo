"""Primary OCR engine: EasyOCR with Arabic + English support."""

from typing import Any, Dict, Optional

from services.ocr_engines.base_ocr_engine import BaseOCREngine


class EasyOCREngine(BaseOCREngine):
    engine_name = "easyocr"

    def __init__(self) -> None:
        self._reader: Optional[Any] = None
        self._load_error: Optional[str] = None

    def _get_reader(self):
        """Lazily create the EasyOCR reader (model download/init is slow)."""
        if self._reader is None and self._load_error is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(["ar", "en"], gpu=False)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._load_error = str(exc)
        return self._reader

    def is_available(self) -> bool:
        return self._get_reader() is not None

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        reader = self._get_reader()
        if reader is None:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": self._load_error or "EasyOCR reader not available",
            }

        try:
            results = reader.readtext(image_path, detail=1)
        except Exception as exc:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"EasyOCR extraction failed: {exc}",
            }

        if not results:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": "EasyOCR returned no text",
            }

        lines = [text for (_bbox, text, _conf) in results]
        confidences = [conf for (_bbox, _text, conf) in results]

        return {
            "text": "\n".join(lines),
            "average_confidence": round(sum(confidences) / len(confidences), 3),
            "status": "success",
            "error": None,
        }
