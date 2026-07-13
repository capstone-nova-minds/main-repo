"""Third OCR engine (voting member): PaddleOCR with Arabic support.

Never replaces EasyOCR -- the OCR router (ocr_router_service.py) runs
all available engines on a page and keeps whichever result scores
highest on calculate_ocr_quality(). If PaddleOCR isn't installed or its
model fails to load, is_available() returns False and the router simply
skips it, exactly like the Tesseract fallback already does.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from services.ocr_engines.base_ocr_engine import BaseOCREngine

ENABLE_PADDLEOCR = os.getenv("ENABLE_PADDLEOCR", "false").lower() == "true"

# Same reasoning as EasyOCR's row-grouping tolerance: PaddleOCR also
# returns detected text boxes in an order that isn't guaranteed to match
# visual reading order, especially for documents with many short,
# closely-spaced fragments (labels, separators, IDs). Grouping by row
# (scaled to the detected text's own size, not a fixed pixel count) then
# ordering right-to-left within each row reconstructs correct Arabic
# reading order regardless of image resolution.
_ROW_GROUPING_TOLERANCE_RATIO = 0.6


def _box_vertical_center(bbox) -> float:
    ys = [point[1] for point in bbox]
    return (min(ys) + max(ys)) / 2


def _box_left_x(bbox) -> float:
    return min(point[0] for point in bbox)


def _box_height(bbox) -> float:
    ys = [point[1] for point in bbox]
    return max(ys) - min(ys)


def _sort_results_by_reading_order(
    results: List[Tuple[Any, str, float]]
) -> List[Tuple[Any, str, float]]:
    """Reorder raw detections into top-to-bottom, right-to-left order."""
    if not results:
        return results

    heights = sorted(_box_height(r[0]) for r in results)
    median_height = heights[len(heights) // 2] if heights else 20.0
    row_tolerance = max(median_height * _ROW_GROUPING_TOLERANCE_RATIO, 6.0)

    by_y = sorted(results, key=lambda r: _box_vertical_center(r[0]))

    rows: List[List[Tuple[Any, str, float]]] = []
    current_row: List[Tuple[Any, str, float]] = []
    current_row_y: Optional[float] = None

    for item in by_y:
        y = _box_vertical_center(item[0])

        if current_row_y is None or abs(y - current_row_y) <= row_tolerance:
            current_row.append(item)
            current_row_y = y if current_row_y is None else (current_row_y + y) / 2
        else:
            rows.append(current_row)
            current_row = [item]
            current_row_y = y

    if current_row:
        rows.append(current_row)

    ordered: List[Tuple[Any, str, float]] = []

    for row in rows:
        row_sorted = sorted(row, key=lambda r: -_box_left_x(r[0]))
        ordered.extend(row_sorted)

    return ordered


class PaddleOCREngineStub(BaseOCREngine):
    # Class name kept as "...Stub" so the existing import in
    # ocr_router_service.py (`from ...paddleocr_engine_stub import
    # PaddleOCREngineStub`) doesn't need to change anywhere else.
    engine_name = "paddleocr"

    def __init__(self) -> None:
        self._ocr: Optional[Any] = None
        self._load_error: Optional[str] = None

    def _get_ocr(self):
        """Lazily create the PaddleOCR reader (model download/init is slow)."""
        if not ENABLE_PADDLEOCR:
            self._load_error = "PaddleOCR disabled via ENABLE_PADDLEOCR=false"
            return None

        if self._ocr is None and self._load_error is None:
            try:
                from paddleocr import PaddleOCR
                from utils.gpu_utils import gpu_available

                use_gpu = gpu_available()

                try:
                    self._ocr = PaddleOCR(use_angle_cls=True, lang="ar", show_log=False, use_gpu=use_gpu)
                except Exception:
                    if use_gpu:
                        self._ocr = PaddleOCR(use_angle_cls=True, lang="ar", show_log=False, use_gpu=False)
                    else:
                        raise
            except Exception as exc:  # pragma: no cover - environment dependent
                self._load_error = str(exc)
        return self._ocr

    def is_available(self) -> bool:
        return self._get_ocr() is not None

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        ocr = self._get_ocr()
        if ocr is None:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": self._load_error or "PaddleOCR reader not available",
            }

        try:
            raw_result = ocr.ocr(image_path, cls=True)
        except Exception as exc:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": f"PaddleOCR extraction failed: {exc}",
            }

        # PaddleOCR's .ocr() returns a list with one entry per page; each
        # entry is a list of [bbox, (text, confidence)].
        page_result = raw_result[0] if raw_result else None

        if not page_result:
            return {
                "text": "",
                "average_confidence": 0.0,
                "status": "failed",
                "error": "PaddleOCR returned no text",
            }

        results: List[Tuple[Any, str, float]] = [
            (bbox, text_conf[0], float(text_conf[1]))
            for bbox, text_conf in page_result
        ]

        results = _sort_results_by_reading_order(results)

        lines = [text for (_bbox, text, _conf) in results]
        confidences = [conf for (_bbox, _text, conf) in results]

        return {
            "text": "\n".join(lines),
            "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "status": "success",
            "error": None,
        }
