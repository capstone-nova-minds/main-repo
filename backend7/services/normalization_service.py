"""Text normalization step between OCR and rule-based extraction."""

from utils.arabic_normalizer import normalize_text


def normalize_ocr_text(raw_text: str) -> dict:
    """Wraps utils.arabic_normalizer.normalize_text for the pipeline.

    Returns {"original_text": ..., "cleaned_text": ...} -- extraction
    services should always work on cleaned_text.
    """
    return normalize_text(raw_text)
