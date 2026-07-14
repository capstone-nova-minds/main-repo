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
    st.markdown("#### Choose one or more files")
    st.caption("Accepted formats: PDF, JPG, JPEG, PNG. Select multiple files to upload and process them together.")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files:
        total_size_kb = sum(f.size for f in uploaded_files) / 1024
        st.caption(f"Selected: **{len(uploaded_files)} file(s)** ({total_size_kb:.1f} KB total)")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

        if st.button(f"📤  Upload {len(uploaded_files)} file(s)", type="primary"):
            documents = []
            errors = []
            progress = st.progress(0.0)
            status_area = st.empty()

            for index, file in enumerate(uploaded_files):
                status_area.info(f"Uploading {file.name}... ({index + 1}/{len(uploaded_files)})")
                try:
                    result = upload_document(file)
                    documents.append({"document_id": result["document_id"], "filename": result["filename"]})
                except Exception as exc:
                    errors.append(f"{file.name}: {exc}")
                progress.progress((index + 1) / len(uploaded_files))

            status_area.empty()
            progress.empty()

            if documents:
                st.session_state["documents"] = documents
                # Singular keys kept for backward compatibility with pages
                # that operate on one "active" document at a time.
                st.session_state["document_id"] = documents[0]["document_id"]
                st.session_state["filename"] = documents[0]["filename"]
                st.session_state.pop("extraction_result", None)
                st.session_state["extraction_results"] = {}
                st.success(f"Uploaded {len(documents)} document(s) successfully.")
                st.info("Next step: open **2 Processing Status** in the sidebar to process them.")

            if errors:
                st.error("Some files failed to upload:")
                for err in errors:
                    st.write(f"- {err}")

if st.session_state.get("documents"):
    st.write("")
    with st.container(border=True):
        st.markdown(f"#### 📄 {len(st.session_state['documents'])} document(s) in session")
        for doc in st.session_state["documents"]:
            st.write(f"**{doc['filename']}** — `{doc['document_id']}`")
