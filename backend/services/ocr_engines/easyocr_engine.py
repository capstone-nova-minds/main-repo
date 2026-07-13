"""Primary OCR engine: EasyOCR with Arabic + English support."""

from typing import Any, Dict, List, Optional, Tuple

from services.ocr_engines.base_ocr_engine import BaseOCREngine

# A word/line is only considered to be on the "same row" as another if
# their vertical centers are within this many pixels of each other.
# Word-level EasyOCR boxes for a normal line of text are usually well
# under this apart; this mainly protects against two genuinely
# different lines (e.g. wrapped rows) getting merged into one.
# A word/line is only considered to be on the "same row" as another if
# their vertical centers are within this fraction of the median detected
# box height. Using a fraction of box height (not a fixed pixel count)
# keeps this correct regardless of image resolution or upscaling -- a
# fixed pixel tolerance broke on the 3x-upscaled header crop, where the
# same visual row's boxes end up further apart in pixel terms, causing
# words on one line (e.g. a city name right after "محكمة صلح") to be
# wrongly split into separate reconstructed rows.
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
    """Reorder raw EasyOCR results into natural top-to-bottom,
    right-to-left (Arabic) reading order.

    EasyOCR returns detections in whatever internal order its detector
    found them in -- NOT necessarily visual reading order. For documents
    with many short, closely-spaced text fragments (labels, separators,
    IDs), trusting that raw order scrambles word order badly (words from
    completely different parts of the page end up adjacent in the
    output text). Sorting by (row, then right-to-left position within
    the row) reconstructs the actual reading order instead.
    """
    if not results:
        return results

    # Row-grouping tolerance scales with the text's own detected size
    # (median box height) instead of a fixed pixel count, so this works
    # the same way on a normal full-page scan and on the 3x-upscaled
    # header crop.
    heights = sorted(_box_height(r[0]) for r in results)
    median_height = heights[len(heights) // 2] if heights else 20.0
    row_tolerance = max(median_height * _ROW_GROUPING_TOLERANCE_RATIO, 6.0)

    # Sort by vertical position first to get a stable top-to-bottom base.
    by_y = sorted(results, key=lambda r: _box_vertical_center(r[0]))

    # Group into rows: consecutive items whose vertical centers are
    # close together belong to the same visual line.
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

    # Within each row, order right-to-left (Arabic reading direction).
    ordered: List[Tuple[Any, str, float]] = []

    for row in rows:
        row_sorted = sorted(row, key=lambda r: -_box_left_x(r[0]))
        ordered.extend(row_sorted)

    return ordered


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
                from utils.gpu_utils import gpu_available

                use_gpu = gpu_available()

                try:
                    self._reader = easyocr.Reader(["ar", "en"], gpu=use_gpu)
                except Exception:
                    # GPU init can fail even when torch.cuda.is_available()
                    # said yes (e.g. out of VRAM, driver mismatch) --
                    # never let that take down OCR entirely, just retry
                    # on CPU.
                    if use_gpu:
                        self._reader = easyocr.Reader(["ar", "en"], gpu=False)
                    else:
                        raise
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

        # Reconstruct visual reading order -- EasyOCR's raw return order
        # is NOT reliable reading order (see _sort_results_by_reading_order).
        results = _sort_results_by_reading_order(results)

        lines = [text for (_bbox, text, _conf) in results]
        # EasyOCR returns numpy.float32/64 confidences -- cast to plain
        # Python float so this never leaks a non-JSON-serializable type
        # downstream (quality scoring, needs_review comparisons, etc.).
        confidences = [float(conf) for (_bbox, _text, conf) in results]

        return {
            "text": "\n".join(lines),
            "average_confidence": round(sum(confidences) / len(confidences), 3),
            "status": "success",
            "error": None,
        }
