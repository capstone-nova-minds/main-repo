"""Basic OpenCV image preprocessing to improve OCR accuracy.

Kept intentionally simple: grayscale -> resize (if small) -> denoise ->
adaptive threshold. Each step is wrapped so a single failure doesn't
crash the whole pipeline (falls back to the least-processed image).
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from services.file_service import PROCESSED_DIR

MIN_DIMENSION = 1000  # upscale small scans so OCR has more pixels to work with


def _processed_dir(document_id: str) -> Path:
    folder = PROCESSED_DIR / document_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _resize_if_small(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    smallest_side = min(height, width)
    if smallest_side >= MIN_DIMENSION:
        return image
    scale = MIN_DIMENSION / smallest_side
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Run the grayscale/resize/denoise/threshold pipeline on one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _resize_if_small(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresholded = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    return thresholded


def preprocess_pages(document_id: str, page_paths: List[Path]) -> List[Path]:
    """Preprocess every page image; on per-page failure, keep the original image."""
    folder = _processed_dir(document_id)
    processed_paths: List[Path] = []

    for page_path in page_paths:
        out_path = folder / page_path.name
        try:
            processed = preprocess_image(page_path)
            cv2.imwrite(str(out_path), processed)
        except Exception:
            # Preprocessing must never crash the pipeline -- fall back to
            # the raw page image so OCR still has something to work with.
            import shutil
            shutil.copyfile(page_path, out_path)
        processed_paths.append(out_path)

    return processed_paths
