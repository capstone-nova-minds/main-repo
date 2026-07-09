"""Page 3: Human review and correction of extracted data (mandatory before export)."""

import streamlit as st

from utils.api_client import get_result, save_review
from components.document_viewer import render_document_viewer
from components.extracted_fields_form import render_document_fields_form
from components.persons_table import render_persons_table

st.title("3. Review & Validation")

if "document_id" not in st.session_state:
    st.warning("No document uploaded yet. Go to the **1_Upload** page first.")
    st.stop()

document_id = st.session_state["document_id"]

if "extraction_result" not in st.session_state:
    with st.spinner("Loading extracted result..."):
        try:
            st.session_state["extraction_result"] = get_result(document_id)
        except Exception as exc:
            st.error(f"Could not load result. Have you run processing yet? ({exc})")
            st.stop()

result = st.session_state["extraction_result"]

col_doc, col_view = st.columns([2, 1])

with col_doc:
    st.subheader("Document Fields")
    edited_document = render_document_fields_form(result.get("document", {}))

    st.subheader("Persons / Companies")
    edited_persons = render_persons_table(result.get("persons", []))

with col_view:
    st.subheader("Original Document")
    render_document_viewer(document_id)

st.divider()

if st.button("Save Reviewed Result"):
    reviewed_payload = {
        "document_id": document_id,
        "document": edited_document,
        "persons": edited_persons,
        "ocr_summary": result.get("ocr_summary", {}),
        "ner_summary": result.get("ner_summary", {}),
    }
    try:
        save_review(document_id, reviewed_payload)
        st.session_state["reviewed_result"] = reviewed_payload
        st.success("Reviewed result saved. You can now proceed to the **4_Export** page.")
    except Exception as exc:
        st.error(f"Failed to save reviewed result: {exc}")
