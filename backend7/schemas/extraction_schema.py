"""Pydantic models for OCR/NER summaries and the final extraction result."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from schemas.document_schema import DocumentInfo
from schemas.person_schema import PersonRecord


class OCRSummary(BaseModel):
    pages: int = 0
    selected_engine: Optional[str] = None
    ocr_status: str = "failed"  # "success" or "failed"
    average_confidence: float = 0.0
    quality_score: float = 0.0
    fallback_used: bool = False


class NERSummary(BaseModel):
    selected_engine: Optional[str] = None
    ner_status: str = "failed"  # "success" or "failed"
    entities_found: int = 0
    person_entities: int = 0
    organization_entities: int = 0
    needs_review: bool = False
    error: Optional[str] = None
    # NER PERSON entities with no nearby National ID and no match to a
    # rule-based record: not trusted enough for final persons, kept here
    # for visibility/debugging only. See entity_merge_service.
    suggested_entities: List[Dict[str, Any]] = []


class ExtractionResult(BaseModel):
    """Full structured output of the extraction pipeline for one document."""

    document_id: str
    document: DocumentInfo = DocumentInfo()
    persons: List[PersonRecord] = []
    ocr_summary: OCRSummary = OCRSummary()
    ner_summary: NERSummary = NERSummary()
    reviewed: bool = False
