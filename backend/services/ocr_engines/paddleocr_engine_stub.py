"""Local PaddleOCR Arabic engine.

The filename and class name still contain "Stub" to preserve compatibility
with the existing OCR router, but this is now a real PaddleOCR engine.

No cloud AI API is used. OCR inference runs locally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.ocr_engines.base_ocr_engine import BaseOCREngine


class PaddleOCREngineStub(BaseOCREngine):
    """Real local PaddleOCR implementation."""

    engine_name = "paddleocr"

    def __init__(self) -> None:
        self._ocr: Optional[Any] = None
        self._load_error: Optional[str] = None
        # Some environments load PaddleOCR fine but its inference call
        # always throws the same environment-level error (e.g. a
        # PIR/oneDNN incompatibility) regardless of the image -- once that
        # happens once, every later page/document would otherwise pay for
        # the same doomed predict() call and its full model-load time on
        # first use. Trip this once and skip straight past PaddleOCR for
        # the rest of the process lifetime instead of retrying forever.
        self._runtime_broken_error: Optional[str] = None

    def _load_engine(self) -> bool:
        """Load PaddleOCR once and reuse it for all OCR requests."""

        if self._ocr is not None:
            return True

        if self._load_error is not None:
            return False

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                lang="ar",
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=os.getenv("PADDLEOCR_DEVICE", "cpu"),
            )

            return True

        except Exception as exc:
            self._load_error = (
                f"Failed to load PaddleOCR: "
                f"{type(exc).__name__}: {exc}"
            )
            self._ocr = None
            return False

    def is_available(self) -> bool:
        """Check whether PaddleOCR is installed and can be loaded."""

        return self._load_engine()

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        """Convert PaddleOCR arrays and tuples into a Python list."""

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if hasattr(value, "tolist"):
            converted = value.tolist()

            if isinstance(converted, list):
                return converted

            return [converted]

        return [value]

    @staticmethod
    def _get_result_payload(result_item: Any) -> Dict[str, Any]:
        """Read the PaddleOCR result regardless of result object format."""

        if isinstance(result_item, dict):
            payload: Any = result_item
        else:
            payload = getattr(result_item, "json", None)

            if callable(payload):
                payload = payload()

            if payload is None:
                payload = getattr(result_item, "res", None)

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return {}

        if not isinstance(payload, dict):
            return {}

        nested_result = payload.get("res")

        if isinstance(nested_result, dict):
            return nested_result

        return payload

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract Arabic text from one local page image."""

        path = Path(image_path)

        if not path.exists():
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"Image file not found: {image_path}",
            }

        if self._runtime_broken_error is not None:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": (
                    "PaddleOCR skipped -- inference failed unrecoverably "
                    f"earlier this run: {self._runtime_broken_error}"
                ),
            }

        if not self._load_engine():
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": self._load_error or "PaddleOCR is unavailable",
            }

        try:
            prediction_results = self._ocr.predict(str(path))

            extracted_lines: List[str] = []
            confidence_scores: List[float] = []

            for result_item in prediction_results:
                result_data = self._get_result_payload(result_item)

                texts = self._to_list(
                    result_data.get("rec_texts")
                )

                scores = self._to_list(
                    result_data.get("rec_scores")
                )

                for index, raw_text in enumerate(texts):
                    clean_text = str(raw_text or "").strip()

                    if not clean_text:
                        continue

                    extracted_lines.append(clean_text)

                    if index < len(scores):
                        try:
                            score = float(scores[index])
                            score = max(0.0, min(1.0, score))
                            confidence_scores.append(score)
                        except (TypeError, ValueError):
                            pass

            final_text = "\n".join(extracted_lines).strip()

            if not final_text:
                return {
                    "text": "",
                    "average_confidence": 0.0,
                    "status": "failed",
                    "error": "PaddleOCR returned no readable text",
                }

            average_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0.0
            )

            return {
                "text": final_text,
                "average_confidence": round(
                    average_confidence,
                    4,
                ),
                "status": "success",
                "error": None,
            }

        except Exception as exc:
            error_message = f"PaddleOCR extraction failed: {type(exc).__name__}: {exc}"

            # NotImplementedError here is an environment/library
            # incompatibility (e.g. PIR attribute conversion in this
            # Paddle build) -- it doesn't depend on the image, so it will
            # fail identically on every page from here on. Anything else
            # (a bad single image, a transient issue) is retried normally.
            if isinstance(exc, NotImplementedError):
                self._runtime_broken_error = error_message

            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": error_message,
            }