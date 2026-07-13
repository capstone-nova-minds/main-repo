"""Placeholder for a future CAMeL Tools Arabic NER engine.

CAMeL Tools is not implemented yet. This stub exists so ner_service can
be extended later without changing its interface -- implement
extract_entities() using CAMeL Tools' NER API when ready.
"""

from typing import Any, Dict, List

from services.ner_engines.base_ner_engine import BaseNEREngine


class CamelToolsNEREngineStub(BaseNEREngine):
    engine_name = "camel_tools"

    def is_available(self) -> bool:
        # Always False until CAMeL Tools support is actually added.
        return False

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []
