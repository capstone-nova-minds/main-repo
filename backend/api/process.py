"""POST /process/{document_id} -- run the full extraction pipeline.
GET /result/{document_id} -- fetch a previously extracted result.
"""

import json
from typing import List

from fastapi import APIRouter, HTTPException

from services.file_service import find_uploaded_file, OCR_OUTPUTS_DIR, ensure_data_dirs
from services.pdf_service import get_document_pages
from services.preprocessing_service import preprocess_pages
from services.ocr_router_service import run_ocr_router
from services.normalization_service import normalize_ocr_text
from services.document_extraction_service import extract_document_fields
from services.person_extraction_service import extract_person_candidates
from services.ner_service import run_ner
from services.entity_merge_service import merge_rules_and_ner
from services.name_splitting_service import expand_grouped_person_records
from services.validation_service import validate_all
from services.review_service import save_extracted_result, load_extracted_result

router = APIRouter()


def _save_ocr_output(document_id: str, ocr_result: dict) -> None:
    ensure_data_dirs()
    path = OCR_OUTPUTS_DIR / f"{document_id}.json"
    path.write_text(json.dumps(ocr_result, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_ocr_summary(ocr_result: dict) -> dict:
    pages = ocr_result.get("pages", [])
    successful_pages = [p for p in pages if p.get("selected_engine")]

    avg_confidence = (
        sum(p["average_confidence"] for p in successful_pages) / len(successful_pages)
        if successful_pages else 0.0
    )
    avg_quality = (
        sum(p["quality_score"] for p in successful_pages) / len(successful_pages)
        if successful_pages else 0.0
    )
    fallback_used = any(
        attempt["engine"] == "tesseract" and attempt["status"] != "skipped"
        for page in pages
        for attempt in page.get("engine_attempts", [])
    )

    return {
        "pages": len(pages),
        "selected_engine": ocr_result.get("selected_engine"),
        "ocr_status": ocr_result.get("ocr_status"),
        "average_confidence": round(avg_confidence, 3),
        "quality_score": round(avg_quality, 3),
        "fallback_used": fallback_used,
    }


@router.post("/process/{document_id}")
def process_document(document_id: str):
    try:
        upload_path, extension = find_uploaded_file(document_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploaded document not found")

    # --- PDF/image -> page images -----------------------------------
    try:
        page_paths: List = get_document_pages(document_id, upload_path, extension)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare page images: {exc}")

    # --- Preprocessing -------------------------------------------------
    try:
        processed_paths = preprocess_pages(document_id, page_paths)
    except Exception:
        # Preprocessing failures already fall back internally; if this
        # somehow still raises, just OCR the raw pages instead.
        processed_paths = page_paths

    # --- OCR Router / Fallback -----------------------------------------
    ocr_result = run_ocr_router(document_id, processed_paths)
    _save_ocr_output(document_id, ocr_result)

    combined_raw_text = "\n".join(p.get("text", "") for p in ocr_result.get("pages", []))

    # --- Normalization ---------------------------------------------------
    normalized = normalize_ocr_text(combined_raw_text)
    cleaned_text = normalized["cleaned_text"]

    # --- Rule-based document extraction -----------------------------------
    document_fields = extract_document_fields(cleaned_text)

    # --- Rule-based person extraction -------------------------------------
    rule_based_persons = extract_person_candidates(cleaned_text)

    # --- Local Arabic NER (never allowed to crash the pipeline) ------------
    try:
        ner_result = run_ner(cleaned_text)
    except Exception as exc:
        ner_result = {"entities": [], "ner_status": "failed", "selected_engine": None, "error": str(exc)}

    # --- Merge rules + NER ---------------------------------------------
    merged_persons = merge_rules_and_ner(cleaned_text, rule_based_persons, ner_result)

    # --- Split grouped family names --------------------------------------
    expanded_persons = expand_grouped_person_records(merged_persons)

    # --- Validation + confidence -------------------------------------------
    validated_document, validated_persons = validate_all(document_fields, expanded_persons)

    person_entities = sum(1 for e in ner_result.get("entities", []) if e.get("label", "").upper() in {"PER", "PERSON"})
    org_entities = sum(1 for e in ner_result.get("entities", []) if e.get("label", "").upper() in {"ORG", "ORGANIZATION"})

    ner_summary = {
        "selected_engine": ner_result.get("selected_engine"),
        "ner_status": ner_result.get("ner_status"),
        "entities_found": len(ner_result.get("entities", [])),
        "person_entities": person_entities,
        "organization_entities": org_entities,
        "needs_review": ner_result.get("ner_status") != "success",
        "error": ner_result.get("error"),
    }

    result = {
        "document_id": document_id,
        "document": {k: v for k, v in validated_document.model_dump().items()},
        "persons": [p.model_dump() for p in validated_persons],
        "ocr_summary": _build_ocr_summary(ocr_result),
        "ner_summary": ner_summary,
        "reviewed": False,
    }

    save_extracted_result(document_id, result)

    return result


@router.get("/result/{document_id}")
def get_result(document_id: str):
    result = load_extracted_result(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No extracted result found for this document")
    return result
