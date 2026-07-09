"""Page 2: Trigger and monitor the extraction pipeline."""

import streamlit as st

from utils.api_client import process_document

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

    if ocr_summary.get("quality_score", 1.0) < 0.65:
        st.warning("منخفضة جودة قراءة النص، يرجى مراجعة البيانات بعناية.")

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
