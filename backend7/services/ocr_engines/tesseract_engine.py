"""Fallback OCR engine: Tesseract with Arabic + English traineddata."""

from typing import Any, Dict

from services.ocr_engines.base_ocr_engine import BaseOCREngine

TESSERACT_LANG = "ara+eng"


class TesseractEngine(BaseOCREngine):
    engine_name = "tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract
            langs = pytesseract.get_languages(config="")
            return "ara" in langs
        except Exception:
            return False

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"Tesseract dependencies unavailable: {exc}",
            }

        try:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as exc:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"Arabic traineddata may be missing: {exc}",
            }
        except Exception as exc:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"Tesseract extraction failed: {exc}",
            }

        words = []
        confidences = []
        for word, conf in zip(data.get("text", []), data.get("conf", [])):
            word = word.strip()
            if not word:
                continue
            words.append(word)
            try:
                conf_value = float(conf)
                if conf_value >= 0:
                    confidences.append(conf_value / 100.0)
            except (TypeError, ValueError):
                continue

        text = " ".join(words)
        if not text:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": "Tesseract returned no text",
            }

        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        return {
            "text": text,
            "average_confidence": avg_confidence,
            "status": "success",
            "error": None,
        }
