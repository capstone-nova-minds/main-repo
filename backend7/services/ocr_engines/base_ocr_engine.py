"""Common interface every OCR engine (EasyOCR, Tesseract, ...) implements."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseOCREngine(ABC):
    engine_name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine's dependencies/models are ready to use."""
        raise NotImplementedError

    @abstractmethod
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Run OCR on one image and return a result dict with keys:

        text (str), average_confidence (float), status ("success"/"failed"),
        error (str or None).
        """
        raise NotImplementedError
