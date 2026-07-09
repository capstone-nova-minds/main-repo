"""Page 4: Export the approved result to JSON and Excel."""

import streamlit as st

from utils.api_client import export_json_url, export_excel_url, download_export

st.title("4. Export")

if "document_id" not in st.session_state:
    st.warning("No document uploaded yet. Go to the **1_Upload** page first.")
    st.stop()

document_id = st.session_state["document_id"]

st.info(
    "Export uses your reviewed data if it was saved on the Review page. "
    "Otherwise, it falls back to the raw extracted data and is marked as not reviewed."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Prepare JSON Export"):
        try:
            content = download_export(export_json_url(document_id))
            st.download_button(
                "Download JSON",
                data=content,
                file_name=f"{document_id}.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Export failed: {exc}")

with col2:
    if st.button("Prepare Excel Export"):
        try:
            content = download_export(export_excel_url(document_id))
            st.download_button(
                "Download Excel",
                data=content,
                file_name=f"{document_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Export failed: {exc}")
