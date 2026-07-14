"""Page 2: Trigger and monitor the extraction pipeline."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.api_client import process_document
from utils.styling import (
    apply_theme,
    render_sidebar_brand,
    render_header,
    render_stepper,
    status_badge,
)

st.set_page_config(
    page_title="Processing — Court Order Extraction",
    page_icon="⚙️",
    layout="wide",
)

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)
OCR_OUTPUTS_DIR = DATA_DIR / "ocr_outputs"

# ---------------------------------------------------------------------------
# Stakeholder-friendly scoring helpers
# ---------------------------------------------------------------------------

def _field_value(field: dict):
    if not isinstance(field, dict):
        return None
    return field.get("value")


def _field_confidence(field: dict) -> float:
    if not isinstance(field, dict):
        return 0.0
    return float(field.get("confidence") or 0.0)


def _field_needs_review(field: dict) -> bool:
    if not isinstance(field, dict):
        return True
    return bool(field.get("needs_review", True))


def calculate_extraction_score(result: dict) -> dict:
    """
    Stakeholder-friendly auto-extraction score.

    Important:
    This is not ground-truth accuracy.
    It measures extracted required fields + review flags.
    """
    document = result.get("document", {}) or {}
    persons = result.get("persons", []) or []

    checks = []

    required_document_fields = [
        ("Court Name", "court_name"),
        ("Case Number", "case_number"),
        ("Document Number", "document_number"),
        ("Document Date", "document_date"),
    ]

    for label, key in required_document_fields:
        field = document.get(key, {}) or {}
        checks.append(
            {
                "field": label,
                "value": _field_value(field),
                "extracted": bool(_field_value(field)),
                "needs_review": _field_needs_review(field),
                "confidence": _field_confidence(field),
            }
        )

    for index, person in enumerate(persons, start=1):
        person_needs_review = bool(person.get("needs_review", True))
        person_confidence = float(person.get("confidence") or 0.0)

        checks.append(
            {
                "field": f"Person {index} Full Name",
                "value": person.get("full_name"),
                "extracted": bool(person.get("full_name")),
                "needs_review": person_needs_review,
                "confidence": person_confidence,
            }
        )

        # A Company record's identifier is its registration_number, not a
        # National ID -- checking national_id here would always fail a
        # correctly-extracted company and unfairly drag the score down.
        is_company = person.get("person_type") == "Company"
        identifier_label = "Registration Number" if is_company else "National ID"
        identifier_value = person.get("registration_number") if is_company else person.get("national_id")

        checks.append(
            {
                "field": f"Person {index} {identifier_label}",
                "value": identifier_value,
                "extracted": bool(identifier_value),
                "needs_review": person_needs_review,
                "confidence": person_confidence,
            }
        )

        checks.append(
            {
                "field": f"Person {index} Type",
                "value": person.get("person_type"),
                "extracted": bool(person.get("person_type")),
                "needs_review": person_needs_review,
                "confidence": person_confidence,
            }
        )

    total_fields = len(checks)
    extracted_fields = sum(1 for item in checks if item["extracted"])
    review_fields = sum(1 for item in checks if item["needs_review"])
    successful_fields = sum(
        1
        for item in checks
        if item["extracted"] and not item["needs_review"]
    )

    score = round((successful_fields / total_fields) * 100, 1) if total_fields else 0.0

    return {
        "score": score,
        "total_fields": total_fields,
        "extracted_fields": extracted_fields,
        "successful_fields": successful_fields,
        "review_fields": review_fields,
        "persons_found": len(persons),
        "checks": checks,
    }


def build_document_table(result: dict) -> pd.DataFrame:
    document = result.get("document", {}) or {}

    rows = []

    field_map = [
        ("Court Name", "court_name"),
        ("Case Number", "case_number"),
        ("Document Number", "document_number"),
        ("Document Date", "document_date"),
    ]

    for label, key in field_map:
        field = document.get(key, {}) or {}
        rows.append(
            {
                "Field": label,
                "Extracted Value": field.get("value"),
                "Confidence": field.get("confidence"),
                "Needs Review": field.get("needs_review"),
            }
        )

    return pd.DataFrame(rows)


def build_persons_table(result: dict) -> pd.DataFrame:
    persons = result.get("persons", []) or []

    rows = []

    for index, person in enumerate(persons, start=1):
        rows.append(
            {
                "#": index,
                "Full Name": person.get("full_name"),
                "National ID": person.get("national_id"),
                "Type": person.get("person_type"),
                "Confidence": person.get("confidence"),
                "Needs Review": person.get("needs_review"),
                "Source": person.get("source"),
                "Extraction Method": person.get("extraction_method"),
            }
        )

    return pd.DataFrame(rows)


def build_field_status_table(result: dict) -> pd.DataFrame:
    score_data = calculate_extraction_score(result)

    rows = []

    for item in score_data["checks"]:
        rows.append(
            {
                "Field": item["field"],
                "Value": item["value"],
                "Extracted": "Yes" if item["extracted"] else "No",
                "Needs Review": "Yes" if item["needs_review"] else "No",
                "Confidence": item["confidence"],
            }
        )

    return pd.DataFrame(rows)


def render_extraction_score(result: dict):
    score_data = calculate_extraction_score(result)

    st.subheader("🎯 Extraction Quality Score")

    st.caption(
        "This score reflects field completeness, confidence, and review requirements. "
        "It is not measured accuracy against ground-truth data."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Quality Score", f"{score_data['score']}%")

    with col2:
        st.metric(
            "Required Fields Extracted",
            f"{score_data['extracted_fields']} / {score_data['total_fields']}",
        )

    with col3:
        st.metric("Fields Need Review", score_data["review_fields"])

    with col4:
        st.metric("Persons Found", score_data["persons_found"])

    progress_value = min(max(score_data["score"] / 100, 0), 1)
    st.progress(progress_value)

    if score_data["score"] >= 90 and score_data["review_fields"] == 0:
        st.success("High extraction quality. The document is ready for review/export.")
    elif score_data["score"] >= 75:
        st.warning("Medium extraction quality. Please review highlighted fields before export.")
    else:
        st.error("Low extraction quality. Manual review is required.")

    st.info(
        "📏 Measured Accuracy will be available after Human Review is saved. "
        "Open **3 Review & Validation** to complete it."
    )

    return score_data


def render_clean_results(result: dict):
    st.write("")
    st.subheader("📋 Extracted Document Information")

    document_df = build_document_table(result)
    st.dataframe(document_df, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("👥 Extracted Persons")

    persons_df = build_persons_table(result)

    if persons_df.empty:
        st.warning("No persons were extracted from this document.")
    else:
        st.dataframe(persons_df, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("✅ Field Status Summary")

    field_df = build_field_status_table(result)
    st.dataframe(field_df, use_container_width=True, hide_index=True)


def render_technical_details(result: dict, document_id: str):
    ocr_summary = result.get("ocr_summary", {}) or {}
    ner_summary = result.get("ner_summary", {}) or {}

    with st.expander("🛠️ Advanced Technical Details", expanded=False):
        st.markdown("### 🔎 OCR Summary")

        ocr_status_tone = "success" if ocr_summary.get("ocr_status") == "success" else "error"
        st.markdown(
            status_badge(
                f"Status: {ocr_summary.get('ocr_status', 'unknown')}",
                ocr_status_tone,
            ),
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected OCR Engine", ocr_summary.get("selected_engine") or "N/A")
        col2.metric("Fallback Used", "Yes" if ocr_summary.get("fallback_used") else "No")
        col3.metric(
            "Average OCR Confidence",
            f"{ocr_summary.get('average_confidence', 0.0):.2f}",
        )

        st.metric(
            "OCR Quality Score",
            f"{ocr_summary.get('quality_score', 0.0):.2f}",
        )

        with st.expander("📄 View OCR Text", expanded=False):

            ocr_output_path = OCR_OUTPUTS_DIR / f"{document_id}.json"

            if ocr_output_path.exists():
                try:
                    ocr_debug = json.loads(
                        ocr_output_path.read_text(encoding="utf-8")
                    )

                    st.caption(
                        f"Lines: {ocr_debug.get('lines_count', 'N/A')} | "
                        f"Fallback used: {'Yes' if ocr_debug.get('fallback_used') else 'No'}"
                    )

                    if ocr_debug.get("header_text"):
                        st.markdown("**Header crop OCR text:**")
                        st.text(ocr_debug.get("header_text", ""))

                    st.markdown("**Full page OCR text:**")
                    st.text(
                        ocr_debug.get("full_page_text")
                        or ocr_debug.get("full_text")
                        or ""
                    )

                    if ocr_debug.get("combined_text"):
                        st.markdown("**Combined OCR text:**")
                        st.text(ocr_debug.get("combined_text", ""))

                except Exception as exc:
                    st.info(f"Could not read OCR debug output: {exc}")
            else:
                st.info("No OCR debug output found for this document yet.")

        st.markdown("---")
        st.markdown("### 🧠 NER Summary")

        ner_tone = "success" if ner_summary.get("ner_status") == "success" else "warning"
        st.markdown(
            status_badge(
                f"Status: {ner_summary.get('ner_status', 'unknown')}",
                ner_tone,
            ),
            unsafe_allow_html=True,
        )

        col6, col7, col8 = st.columns(3)
        col6.metric("Selected NER Engine", ner_summary.get("selected_engine") or "N/A")
        col7.metric("Entities Found", ner_summary.get("entities_found", 0))
        col8.metric("Person Entities", ner_summary.get("person_entities", 0))

        col9, col10 = st.columns(2)
        col9.metric("Organization Entities", ner_summary.get("organization_entities", 0))
        col10.metric(
            "Suggested Unconfirmed",
            len(ner_summary.get("suggested_entities", []) or []),
        )

        if ner_summary.get("ner_status") != "success":
            st.info(
                f"NER unavailable: "
                f"{ner_summary.get('error') or 'continuing with rule-based extraction only.'}"
            )

        st.markdown("---")
        st.markdown("### 🧾 Extracted JSON Preview")
        st.json(result)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

apply_theme()
render_sidebar_brand()
render_header(
    "⚙️",
    "Processing Status",
    "Run OCR, local Arabic NER, and rule-based extraction on the uploaded document(s).",
)
render_stepper(2)

documents = st.session_state.get("documents")
if not documents:
    # Backward compatibility with a single document uploaded in an older session shape.
    if "document_id" in st.session_state:
        documents = [{
            "document_id": st.session_state["document_id"],
            "filename": st.session_state.get("filename", "unknown"),
        }]
    else:
        st.warning("No document uploaded yet. Open **1 Upload** in the sidebar first.")
        st.stop()

st.session_state.setdefault("extraction_results", {})
results = st.session_state["extraction_results"]

with st.container(border=True):
    st.markdown(f"**{len(documents)} document(s) ready to process:**")
    for doc in documents:
        mark = "✅" if doc["document_id"] in results else "⏳"
        st.write(f"{mark} {doc['filename']} — `{doc['document_id']}`")

    start_clicked = st.button(
        f"▶️  Process All ({len(documents)})",
        type="primary",
        use_container_width=True,
    )

if start_clicked:
    progress = st.progress(0.0)
    status_area = st.empty()
    errors = []

    for index, doc in enumerate(documents):
        status_area.info(
            f"Processing {doc['filename']}... ({index + 1}/{len(documents)}) "
            "this can take a while on CPU."
        )
        try:
            results[doc["document_id"]] = process_document(doc["document_id"])
        except Exception as exc:
            errors.append(f"{doc['filename']}: {exc}")
        progress.progress((index + 1) / len(documents))

    status_area.empty()
    progress.empty()

    if errors:
        st.error("Some documents failed to process:")
        for err in errors:
            st.write(f"- {err}")
    else:
        st.success(f"Processed {len(documents)} document(s).")

if results:
    st.write("")
    st.subheader("📊 Summary — all processed documents")

    summary_rows = []
    for doc in documents:
        result = results.get(doc["document_id"])
        if result is None:
            summary_rows.append({
                "Filename": doc["filename"],
                "Document ID": doc["document_id"][:8] + "…",
                "Status": "Not processed yet",
                "Quality Score": None,
                "Persons Found": None,
            })
            continue
        score_data = calculate_extraction_score(result)
        summary_rows.append({
            "Filename": doc["filename"],
            "Document ID": doc["document_id"][:8] + "…",
            "Status": "Processed",
            "Quality Score": f"{score_data['score']}%",
            "Persons Found": score_data["persons_found"],
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Default the "active" document (used by Review & Export) to the first
    # one, but each page also has its own selector so this is just a sane
    # starting point, not something the reviewer needs to manage here.
    st.session_state.setdefault("document_id", documents[0]["document_id"])
    st.session_state.setdefault("filename", documents[0]["filename"])

    st.write("")
    st.subheader("🔍 Details — click a tab to see everything for that document")

    tab_labels = [f"{doc['filename']}" for doc in documents]
    tabs = st.tabs(tab_labels)

    for tab, doc in zip(tabs, documents):
        with tab:
            doc_id = doc["document_id"]
            result = results.get(doc_id)

            if result is None:
                st.info("This document hasn't been processed yet. Click **Process All** above.")
                continue

            render_extraction_score(result)
            render_clean_results(result)

            st.write("")
            st.info("➡️ Next step: open **3 Review & Validation** in the sidebar to review and correct the extracted data.")

            st.write("")
            render_technical_details(result, doc_id)