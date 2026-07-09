"""Page 1: Upload a court order document."""

import streamlit as st

from utils.api_client import upload_document

st.title("1. Upload Court Order")

st.write("Upload a scanned court attachment order (PDF, JPG, JPEG, or PNG).")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Upload"):
        with st.spinner("Uploading document..."):
            try:
                result = upload_document(uploaded_file)
                st.session_state["document_id"] = result["document_id"]
                st.session_state["filename"] = result["filename"]
                st.success(f"Uploaded successfully: {result['filename']}")
                st.write(f"**Document ID:** `{result['document_id']}`")
                st.info("Next step: go to the **2_Processing_Status** page to start processing.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

if "document_id" in st.session_state:
    st.divider()
    st.write("Current document in session:")
    st.write(f"- **Document ID:** `{st.session_state['document_id']}`")
    st.write(f"- **Filename:** {st.session_state.get('filename', 'unknown')}")
