"""Recursively convert numpy/pathlib types into plain JSON-serializable
Python types before json.dumps() or returning a FastAPI response.

EasyOCR/OpenCV/numpy computations (confidence scores, quality scores,
comparisons derived from them) can leak numpy.float32/64, numpy.int64,
numpy.bool_, and numpy.ndarray into otherwise-plain dicts. Those types
look like normal Python values but json.dumps() rejects them, so every
save/return point in the pipeline should pass through this function.
"""

from pathlib import Path

import numpy as np


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
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    return obj
