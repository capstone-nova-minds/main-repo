"""Page 2: Trigger and monitor the extraction pipeline."""

import json
import os
from pathlib import Path

import streamlit as st

from utils.api_client import process_document

# Backend and frontend share the same data/ folder (a Docker volume in
# containers, or the repo-root data/ folder when run locally) -- same
# pattern as components/document_viewer.py.
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
OCR_OUTPUTS_DIR = DATA_DIR / "ocr_outputs"

OCR_QUALITY_WARNING_THRESHOLD = 0.75

st.title("2. Processing Status")

if "document_id" not in st.session_state:
    st.warning("No document uploaded yet. Go to the **1_Upload** page first.")
    st.stop()

document_id = st.session_state["document_id"]
st.write(f"**Document ID:** `{document_id}`")

if st.button("Start Processing"):
    with st.spinner("Running OCR, NER, and rule-based extraction... this can take a while on CPU."):
        try:
            result = process_document(document_id)
            st.session_state["extraction_result"] = result
            st.success("Processing complete.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.stop()

if "extraction_result" in st.session_state:
    result = st.session_state["extraction_result"]
    ocr_summary = result.get("ocr_summary", {})
    ner_summary = result.get("ner_summary", {})

    st.subheader("OCR Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Selected OCR Engine", ocr_summary.get("selected_engine") or "N/A")
    col2.metric("OCR Status", ocr_summary.get("ocr_status", "unknown"))
    col3.metric("Fallback Used", "Yes" if ocr_summary.get("fallback_used") else "No")

    col4, col5 = st.columns(2)
    col4.metric("Average OCR Confidence", f"{ocr_summary.get('average_confidence', 0.0):.2f}")
    col5.metric("OCR Quality Score", f"{ocr_summary.get('quality_score', 0.0):.2f}")

    if ocr_summary.get("quality_score", 1.0) < OCR_QUALITY_WARNING_THRESHOLD:
        st.warning("منخفضة جودة قراءة النص، يرجى مراجعة البيانات بعناية.")

    with st.expander("View OCR Text"):
        ocr_output_path = OCR_OUTPUTS_DIR / f"{document_id}.json"
        if ocr_output_path.exists():
            try:
                ocr_debug = json.loads(ocr_output_path.read_text(encoding="utf-8"))
                st.caption(
                    f"Lines: {ocr_debug.get('lines_count', 'N/A')} | "
                    f"Fallback used: {'Yes' if ocr_debug.get('fallback_used') else 'No'}"
                )
                if ocr_debug.get("header_text"):
                    st.markdown("**Header crop OCR text:**")
                    st.text(ocr_debug["header_text"])
                st.markdown("**Full page OCR text:**")
                st.text(ocr_debug.get("full_text", ""))
            except Exception as exc:
                st.info(f"Could not read OCR debug output: {exc}")
        else:
            st.info("No OCR debug output found for this document yet.")

    st.subheader("NER Summary")
    col6, col7, col8 = st.columns(3)
    col6.metric("NER Status", ner_summary.get("ner_status", "unknown"))
    col7.metric("Selected NER Engine", ner_summary.get("selected_engine") or "N/A")
    col8.metric("Entities Found", ner_summary.get("entities_found", 0))

    col9, col10 = st.columns(2)
    col9.metric("Person Entities", ner_summary.get("person_entities", 0))
    col10.metric("Organization Entities", ner_summary.get("organization_entities", 0))

    if ner_summary.get("ner_status") != "success":
        st.info(f"NER unavailable: {ner_summary.get('error') or 'continuing with rule-based extraction only.'}")

    st.subheader("Extracted JSON Preview")
    st.json(result)

    st.info("Next step: go to the **3_Review_Validation** page to review and correct the extracted data.")
