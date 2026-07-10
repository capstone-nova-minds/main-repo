"""POST /process/{document_id} -- run the full extraction pipeline.
GET /result/{document_id} -- fetch a previously extracted result.
"""

import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from services.file_service import find_uploaded_file, OCR_OUTPUTS_DIR, ensure_data_dirs
from services.pdf_service import get_document_pages
from services.preprocessing_service import preprocess_pages
from services.ocr_router_service import run_ocr_router
from services.header_extraction_service import extract_header_text
from services.normalization_service import normalize_ocr_text
from services.document_extraction_service import extract_document_fields
from services.person_extraction_service import extract_person_candidates, find_national_ids
from services.ner_service import run_ner
from services.entity_merge_service import merge_rules_and_ner
from services.name_splitting_service import expand_grouped_person_records
from services.validation_service import validate_all
from services.review_service import save_extracted_result, load_extracted_result
from utils.json_utils import make_json_serializable

router = APIRouter()

# Debug logging here is metadata-only: counts and booleans,
# never National IDs or raw OCR text.
logger = logging.getLogger(__name__)


def _safe_model_dump(obj):
    """Support Pydantic v1 and v2."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _save_ocr_output(document_id: str, ocr_result: dict) -> None:
    ensure_data_dirs()
    safe_result = make_json_serializable(ocr_result)
    path = OCR_OUTPUTS_DIR / f"{document_id}.json"
    path.write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_ocr_summary(ocr_result: dict) -> dict:
    pages = ocr_result.get("pages", [])
    successful_pages = [p for p in pages if p.get("selected_engine")]

    avg_confidence = (
        sum(p.get("average_confidence", 0.0) for p in successful_pages) / len(successful_pages)
        if successful_pages else 0.0
    )

    avg_quality = (
        sum(p.get("quality_score", 0.0) for p in successful_pages) / len(successful_pages)
        if successful_pages else 0.0
    )

    fallback_used = any(
        attempt.get("engine") == "tesseract" and attempt.get("status") != "skipped"
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


def _deduplicate_person_records(records: list) -> list:
    """Deduplicate person records by national_id first, then full_name."""
    best_by_id = {}
    no_id_records = []
    seen_names = set()

    for record in records:
        national_id = getattr(record, "national_id", None)
        full_name = getattr(record, "full_name", None)

        if national_id:
            if national_id not in best_by_id:
                best_by_id[national_id] = record
                continue

            existing = best_by_id[national_id]
            existing_name_len = len((getattr(existing, "full_name", "") or "").split())
            new_name_len = len((full_name or "").split())

            existing_method = getattr(existing, "extraction_method", None)
            new_method = getattr(record, "extraction_method", None)

            if new_method == "direct_legal_phrase" and existing_method != "direct_legal_phrase":
                best_by_id[national_id] = record
            elif new_name_len > existing_name_len:
                best_by_id[national_id] = record
        else:
            no_id_records.append(record)

    unique_records = list(best_by_id.values())

    for record in unique_records:
        if getattr(record, "full_name", None):
            seen_names.add(record.full_name)

    for record in no_id_records:
        full_name = getattr(record, "full_name", None)

        if not full_name:
            continue

        if full_name in seen_names:
            continue

        seen_names.add(full_name)
        unique_records.append(record)

    return unique_records


def _extract_persons_with_fallbacks(
    cleaned_text: str,
    full_page_text: str,
    combined_text: str,
) -> tuple[list, dict]:
    """
    Try person extraction using multiple text versions.

    Why:
    OCR line breaks sometimes split the phrase:
    للمستدعى ضده + الاسم + رقم وطني

    So we try:
    1. cleaned_text
    2. normalized full_page_text
    3. combined_text collapsed into one line
    """
    attempts = []
    all_records = []

    def run_attempt(label: str, text: str) -> None:
        if not text:
            attempts.append(
                {
                    "label": label,
                    "national_ids_found": 0,
                    "records_found": 0,
                }
            )
            return

        normalized = normalize_ocr_text(text)
        attempt_text = normalized.get("cleaned_text", "")

        national_id_count = len(find_national_ids(attempt_text))
        records = extract_person_candidates(attempt_text)

        attempts.append(
            {
                "label": label,
                "national_ids_found": national_id_count,
                "records_found": len(records),
            }
        )

        all_records.extend(records)

        logger.info(
            "person_extraction_attempt=%s national_ids_found=%d records_found=%d",
            label,
            national_id_count,
            len(records),
        )

    # Main attempt.
    run_attempt("cleaned_text", cleaned_text)

    # Fallback 1: full OCR text only.
    if not all_records:
        run_attempt("full_page_text", full_page_text)

    # Fallback 2: combined text in one line.
    if not all_records:
        combined_one_line = " ".join((combined_text or "").splitlines())
        run_attempt("combined_one_line", combined_one_line)

    final_records = _deduplicate_person_records(all_records)

    debug = {
        "attempts": attempts,
        "final_rule_based_records": len(final_records),
    }

    return final_records, debug


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
        logger.exception("document_id=%s preprocessing_failed_using_raw_pages", document_id)
        processed_paths = page_paths

    # --- OCR Router / Fallback -----------------------------------------
    try:
        ocr_result = run_ocr_router(document_id, processed_paths)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    full_page_text = "\n".join(
        p.get("text", "")
        for p in ocr_result.get("pages", [])
    )

    # --- Header crop OCR -----------------------------------------------
    header_texts = []

    for page_number, page_path in enumerate(processed_paths, start=1):
        try:
            header_result = extract_header_text(page_number, page_path)
        except Exception:
            logger.exception(
                "document_id=%s header_ocr_failed page=%d",
                document_id,
                page_number,
            )
            header_result = {"text": ""}

        if header_result.get("text"):
            header_texts.append(header_result["text"])

    header_text = "\n".join(header_texts)
    combined_text = f"{header_text}\n{full_page_text}" if header_text else full_page_text

    # Save OCR debug output.
    ocr_summary_preview = _build_ocr_summary(ocr_result)
    ocr_result["header_text"] = header_text
    ocr_result["full_page_text"] = full_page_text
    ocr_result["combined_text"] = combined_text
    ocr_result["fallback_used"] = ocr_summary_preview["fallback_used"]
    ocr_result["average_confidence"] = ocr_summary_preview["average_confidence"]
    ocr_result["quality_score"] = ocr_summary_preview["quality_score"]
    ocr_result["lines_count"] = len(full_page_text.splitlines())
    _save_ocr_output(document_id, ocr_result)

    # --- Normalization ---------------------------------------------------
    normalized = normalize_ocr_text(combined_text)
    cleaned_text = normalized.get("cleaned_text", "")

    normalized_header_text = (
        normalize_ocr_text(header_text).get("cleaned_text", "")
        if header_text else None
    )

    logger.info(
        "document_id=%s cleaned_text_lines=%d",
        document_id,
        len(cleaned_text.splitlines()),
    )

    # --- Rule-based document extraction ---------------------------------
    document_fields = extract_document_fields(
        cleaned_text,
        header_text=normalized_header_text,
    )

    document_number_fallback_used = (
        document_fields.document_number.value is not None
        and document_fields.case_number.value is not None
        and document_fields.document_number.value == document_fields.case_number.value
        and document_fields.document_number.needs_review
    )

    logger.info(
        "document_id=%s date_found=%s document_number_fallback_used=%s",
        document_id,
        document_fields.document_date.value is not None,
        document_number_fallback_used,
    )

    # --- Rule-based person extraction -----------------------------------
    national_id_count = len(find_national_ids(cleaned_text))

    rule_based_persons, person_extraction_debug = _extract_persons_with_fallbacks(
        cleaned_text=cleaned_text,
        full_page_text=full_page_text,
        combined_text=combined_text,
    )

    logger.info(
        "document_id=%s national_ids_found=%d rule_based_records=%d",
        document_id,
        national_id_count,
        len(rule_based_persons),
    )

    if national_id_count > 0 and len(rule_based_persons) == 0:
        logger.warning(
            "document_id=%s national_ids_exist_but_no_rule_based_persons",
            document_id,
        )

    # --- Local Arabic NER -----------------------------------------------
    try:
        ner_result = run_ner(cleaned_text)
    except Exception as exc:
        ner_result = {
            "entities": [],
            "ner_status": "failed",
            "selected_engine": None,
            "error": str(exc),
        }

    logger.info(
        "document_id=%s ner_status=%s ner_records=%d",
        document_id,
        ner_result.get("ner_status"),
        len(ner_result.get("entities", [])),
    )

    # --- Merge rules + NER ----------------------------------------------
    merged_persons, suggested_entities = merge_rules_and_ner(
        cleaned_text,
        rule_based_persons,
        ner_result,
    )

    # --- Split grouped family names -------------------------------------
    expanded_persons = expand_grouped_person_records(merged_persons)

    # --- Validation + confidence ----------------------------------------
    validated_document, validated_persons = validate_all(
        document_fields,
        expanded_persons,
    )

    logger.info(
        "document_id=%s final_validated_person_records=%d",
        document_id,
        len(validated_persons),
    )

    person_entities = sum(
        1 for e in ner_result.get("entities", [])
        if e.get("label", "").upper() in {"PER", "PERSON"}
    )

    org_entities = sum(
        1 for e in ner_result.get("entities", [])
        if e.get("label", "").upper() in {"ORG", "ORGANIZATION"}
    )

    ner_summary = {
        "selected_engine": ner_result.get("selected_engine"),
        "ner_status": ner_result.get("ner_status"),
        "entities_found": len(ner_result.get("entities", [])),
        "person_entities": person_entities,
        "organization_entities": org_entities,
        "needs_review": ner_result.get("ner_status") != "success",
        "error": ner_result.get("error"),
        "suggested_entities": suggested_entities,
    }

    extraction_debug = {
        "national_ids_found_in_cleaned_text": national_id_count,
        "rule_based_person_records": len(rule_based_persons),
        "merged_person_records": len(merged_persons),
        "expanded_person_records": len(expanded_persons),
        "validated_person_records": len(validated_persons),
        "person_extraction_attempts": person_extraction_debug.get("attempts", []),
    }

    result = {
        "document_id": document_id,
        "document": {
            k: v for k, v in _safe_model_dump(validated_document).items()
        },
        "persons": [
            _safe_model_dump(p)
            for p in validated_persons
        ],
        "ocr_summary": _build_ocr_summary(ocr_result),
        "ner_summary": ner_summary,
        "extraction_debug": extraction_debug,
        "reviewed": False,
    }

    result = make_json_serializable(result)

    save_extracted_result(document_id, result)

    return result


@router.get("/result/{document_id}")
def get_result(document_id: str):
    result = load_extracted_result(document_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No extracted result found for this document",
        )

    return result