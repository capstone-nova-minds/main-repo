"""Page 1: Upload a court order document."""

import streamlit as st

from utils.api_client import API_BASE_URL, check_backend_health, upload_document
from utils.styling import apply_theme, render_sidebar_brand, render_header, render_stepper, status_badge

st.set_page_config(page_title="Upload — Court Order Extraction", page_icon="📤", layout="wide")

apply_theme()
render_sidebar_brand()
render_header("📤", "Upload Court Order", "Submit a scanned Arabic court attachment order to begin.")
render_stepper(1)

is_healthy, health_message = check_backend_health()

badge_html = status_badge("● Backend online", "success") if is_healthy else status_badge("● Backend unreachable", "error")
st.markdown(f"{badge_html} &nbsp; `{API_BASE_URL}`", unsafe_allow_html=True)
st.write("")

if not is_healthy:
    st.error(f"Backend is not reachable. Please make sure FastAPI is running on {API_BASE_URL}.")
    with st.expander("Technical details"):
        st.write(f"**API_BASE_URL:** `{API_BASE_URL}`")
        st.write(f"**Health check error:** {health_message}")
    st.stop()

with st.container(border=True):
    st.markdown("#### Choose a file")
    st.caption("Accepted formats: PDF, JPG, JPEG, PNG")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.caption(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        if st.button("📤  Upload", type="primary"):
            with st.spinner("Uploading document..."):
                try:
                    result = upload_document(uploaded_file)
                    st.session_state["document_id"] = result["document_id"]
                    st.session_state["filename"] = result["filename"]
                    st.success(f"Uploaded successfully: {result['filename']}")
                    st.info("Next step: open **2 Processing Status** in the sidebar to start processing.")
                except Exception as exc:
                    st.error("Upload failed. See details below.")
                    st.write(f"**API_BASE_URL:** `{API_BASE_URL}`")
                    st.write(f"**Backend health check:** {'passed' if is_healthy else 'failed'}")
                    with st.expander("Technical details"):
                        st.exception(exc)

if "document_id" in st.session_state:
    st.write("")
    with st.container(border=True):
        st.markdown("#### 📄 Current document in session")
        col1, col2 = st.columns(2)
        col1.metric("Document ID", st.session_state["document_id"][:8] + "…")
        col2.metric("Filename", st.session_state.get("filename", "unknown"))
        st.caption(f"Full ID: `{st.session_state['document_id']}`")
