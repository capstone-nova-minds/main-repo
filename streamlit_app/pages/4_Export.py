"""Page 4: Export the approved result to JSON and Excel."""

import streamlit as st

from utils.api_client import export_json_url, export_excel_url, download_export
from utils.styling import apply_theme, render_sidebar_brand, render_header, render_stepper

st.set_page_config(page_title="Export — Court Order Extraction", page_icon="📦", layout="wide")

apply_theme()
render_sidebar_brand()
render_header("📦", "Export", "Download the approved result as JSON or Excel.")
render_stepper(4)

if "document_id" not in st.session_state:
    st.warning("No document uploaded yet. Open **1 Upload** in the sidebar first.")
    st.stop()

document_id = st.session_state["document_id"]

st.info(
    "ℹ️ Export uses your reviewed data if it was saved on the Review page. "
    "Otherwise, it falls back to the raw extracted data and is marked as not reviewed."
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 🗂️ JSON")
        st.caption("Full structured result, including document fields and person records.")
        if st.button("Prepare JSON Export", use_container_width=True):
            try:
                content = download_export(export_json_url(document_id))
                st.download_button(
                    "⬇️  Download JSON",
                    data=content,
                    file_name=f"{document_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

with col2:
    with st.container(border=True):
        st.markdown("### 📊 Excel")
        st.caption("One row per person, with document fields repeated on each row.")
        if st.button("Prepare Excel Export", use_container_width=True):
            try:
                content = download_export(export_excel_url(document_id))
                st.download_button(
                    "⬇️  Download Excel",
                    data=content,
                    file_name=f"{document_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")
