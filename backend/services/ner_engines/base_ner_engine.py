"""Common interface every NER engine (Stanza, CAMeL Tools, ...) implements."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseNEREngine(ABC):
    engine_name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine's dependencies/models are ready to use."""
        raise NotImplementedError

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Return a list of entity dicts: text, label, start_char, end_char,
        confidence, source.
        """
        raise NotImplementedError
