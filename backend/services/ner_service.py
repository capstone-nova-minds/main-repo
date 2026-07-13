"""Runs local Arabic NER over normalized OCR text.

NER supports rule-based extraction -- it never replaces it, and a
failure here must never crash the pipeline (falls back to rules-only).
"""

import os
from typing import Any, Dict

from services.ner_engines.stanza_ner_engine import StanzaNEREngine
from services.ner_engines.camel_tools_ner_engine_stub import CamelToolsNEREngineStub

ENABLE_NER = os.getenv("ENABLE_NER", "true").lower() == "true"

_stanza_engine = StanzaNEREngine()
_camel_tools_engine = CamelToolsNEREngineStub()

# PER/PERSON -> Individual candidates, ORG/ORGANIZATION -> Company candidates.
PERSON_LABELS = {"PER", "PERSON"}
ORG_LABELS = {"ORG", "ORGANIZATION"}


def run_ner(text: str) -> Dict[str, Any]:
    """Run the primary NER engine (Stanza) and return the documented NER output shape."""
    if not ENABLE_NER:
        return {
            "entities": [],
            "ner_status": "failed",
            "selected_engine": None,
            "error": "NER disabled via ENABLE_NER=false",
        }

    try:
        if not _stanza_engine.is_available():
            return {
                "entities": [],
                "ner_status": "failed",
                "selected_engine": None,
                "error": "Stanza Arabic model not available -- continuing with rules only",
            }

        entities = _stanza_engine.extract_entities(text)
        return {
            "entities": entities,
            "ner_status": "success",
            "selected_engine": _stanza_engine.engine_name,
            "error": None,
        }
    except Exception as exc:
        # NER failure must not crash the app -- caller continues with rules.
        return {
            "entities": [],
            "ner_status": "failed",
            "selected_engine": None,
            "error": f"NER failed unexpectedly: {exc}",
        }
