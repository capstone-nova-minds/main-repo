"""Basic OpenCV image preprocessing to improve OCR accuracy.

Kept intentionally simple: grayscale -> upscale (if small) -> light
denoise -> contrast enhancement. Each step is wrapped so a single
failure doesn't crash the whole pipeline (falls back to the
least-processed image).

Note: this intentionally does NOT hard-binarize (cv2.adaptiveThreshold)
the image. Small header text (case number, date) is often only a few
pixels tall, and hard thresholding tends to erase or fuse those thin
strokes -- which is what was hurting OCR quality on document headers.
CLAHE contrast enhancement improves legibility without destroying it.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from services.file_service import PROCESSED_DIR

# If the image is narrower than this, upscale 2x so small header text
# (case number, date) has enough pixels for OCR to resolve.
WIDTH_UPSCALE_THRESHOLD = 1800
UPSCALE_FACTOR = 2


def _processed_dir(document_id: str) -> Path:
    folder = PROCESSED_DIR / document_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _upscale_if_small(image: np.ndarray) -> np.ndarray:
    _height, width = image.shape[:2]
    if width >= WIDTH_UPSCALE_THRESHOLD:
        return image
    return cv2.resize(
        image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )


def preprocess_image(image_path: Path) -> np.ndarray:
    """Run the grayscale/upscale/denoise/contrast pipeline on one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _upscale_if_small(gray)

    # Light denoise only -- a small h keeps thin header strokes intact.
    # Explicit template/search window sizes: the default searchWindowSize
    # (21) makes this the single slowest step in preprocessing on CPU --
    # its cost scales roughly with searchWindowSize squared. 15 keeps
    # denoising effective while meaningfully cutting processing time.
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # CLAHE (contrast-limited adaptive histogram equalization) boosts
    # local contrast so faint/small text becomes easier to read, without
    # collapsing it to hard black/white like adaptiveThreshold does.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


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
