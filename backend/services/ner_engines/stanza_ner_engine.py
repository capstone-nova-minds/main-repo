"""Primary local Arabic NER engine, backed by Stanza.

Runs entirely offline once the 'ar' model is downloaded (the backend
Dockerfile tries to pre-download it at build time). If the model isn't
available, is_available() returns False and the caller falls back to
rule-based extraction only -- NER never blocks the pipeline.
"""

import os
from typing import Any, Dict, List, Optional

from services.ner_engines.base_ner_engine import BaseNEREngine

DEFAULT_CONFIDENCE = float(os.getenv("NER_CONFIDENCE_DEFAULT", "0.75"))


class StanzaNEREngine(BaseNEREngine):
    engine_name = "stanza"

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._load_error: Optional[str] = None

    def _get_pipeline(self):
        """Lazily build the Stanza Arabic pipeline (tokenize + ner only)."""
        if self._pipeline is None and self._load_error is None:
            try:
                import stanza
                from utils.gpu_utils import gpu_available

                use_gpu = gpu_available()

                try:
                    self._pipeline = stanza.Pipeline(
                        lang="ar",
                        processors="tokenize,ner",
                        verbose=False,
                        download_method=None,
                        use_gpu=use_gpu,
                    )
                except Exception:
                    if use_gpu:
                        self._pipeline = stanza.Pipeline(
                            lang="ar",
                            processors="tokenize,ner",
                            verbose=False,
                            download_method=None,
                            use_gpu=False,
                        )
                    else:
                        raise
            except Exception as exc:  # pragma: no cover - environment dependent
                self._load_error = str(exc)
        return self._pipeline

    def is_available(self) -> bool:
        return self._get_pipeline() is not None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        pipeline = self._get_pipeline()
        if pipeline is None or not text or not text.strip():
            return []

        doc = pipeline(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.type,
                "start_char": getattr(ent, "start_char", None),
                "end_char": getattr(ent, "end_char", None),
                "confidence": DEFAULT_CONFIDENCE,
                "source": "stanza_ner",
            })
        return entities
