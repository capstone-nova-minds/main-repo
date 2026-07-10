"""Page 1: Upload a court order document."""

import streamlit as st

from utils.api_client import API_BASE_URL, check_backend_health, upload_document

st.title("1. Upload Court Order")

st.caption(f"Backend API: `{API_BASE_URL}`")

is_healthy, health_message = check_backend_health()
if not is_healthy:
    st.error(
        "Backend is not reachable. Please make sure FastAPI is running on "
        f"{API_BASE_URL}."
    )
    with st.expander("Technical details"):
        st.write(f"**API_BASE_URL:** `{API_BASE_URL}`")
        st.write(f"**Health check error:** {health_message}")
    st.stop()

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
                st.error("Upload failed. See details below.")
                st.write(f"**API_BASE_URL:** `{API_BASE_URL}`")
                st.write(f"**Backend health check:** {'passed' if is_healthy else 'failed'}")
                with st.expander("Technical details"):
                    st.exception(exc)

if "document_id" in st.session_state:
    st.divider()
    st.write("Current document in session:")
    st.write(f"- **Document ID:** `{st.session_state['document_id']}`")
    st.write(f"- **Filename:** {st.session_state.get('filename', 'unknown')}")
