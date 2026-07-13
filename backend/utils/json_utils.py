"""Recursively convert numpy/pathlib types into plain JSON-serializable
Python types before json.dumps() or returning a FastAPI response.

EasyOCR/OpenCV/numpy computations (confidence scores, quality scores,
comparisons derived from them) can leak numpy.float32/64, numpy.int64,
numpy.bool_, and numpy.ndarray into otherwise-plain dicts. Those types
look like normal Python values but json.dumps() rejects them, so every
save/return point in the pipeline should pass through this function.

Float values also get checked for NaN/Infinity here. Those aren't valid
JSON per spec, and FastAPI's JSONResponse explicitly rejects them
(allow_nan=False) -- raising "Out of range float values are not JSON
compliant" the moment one reaches a save/response call, however deep in
a scoring/averaging calculation it originated from (e.g. a stray
division that produced NaN). Converting them to None here means a bad
score gets flagged as missing/needs-review instead of crashing the save.
"""

import math
from pathlib import Path

import numpy as np


def _sanitize_float(value: float):
    """Return None for NaN/Infinity, otherwise the plain float unchanged."""
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def make_json_serializable(obj):
    """Recursively convert obj into something json.dumps() can handle."""
    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return _sanitize_float(float(obj))

    if isinstance(obj, float):
        return _sanitize_float(obj)

    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())

    if isinstance(obj, Path):
        return str(obj)

    return obj