"""Page 4: Export the approved result to JSON and Excel."""

import io
import json

import pandas as pd
import streamlit as st

from utils.api_client import export_json_url, export_excel_url, download_export
from utils.styling import apply_theme, render_sidebar_brand, render_header, render_stepper

st.set_page_config(page_title="Export — Court Order Extraction", page_icon="📦", layout="wide")

apply_theme()
render_sidebar_brand()
render_header("📦", "Export", "Download the approved result as JSON or Excel.")
render_stepper(4)

documents = st.session_state.get("documents", [])
if documents:
    doc_options = {f"{doc['filename']} ({doc['document_id'][:8]}…)": doc["document_id"] for doc in documents}
    labels = list(doc_options.keys())
    default_index = 0
    for i, doc_id in enumerate(doc_options.values()):
        if doc_id == st.session_state.get("document_id"):
            default_index = i
            break
    selected_label = st.selectbox("📄 Exporting document:", labels, index=default_index)
    st.session_state["document_id"] = doc_options[selected_label]
    st.write("")

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

if len(documents) > 1:
    st.write("")
    st.subheader("📦 Export all documents together")
    st.caption(f"Combine all {len(documents)} uploaded documents into a single file.")

    col_all_json, col_all_excel = st.columns(2)

    with col_all_json:
        with st.container(border=True):
            st.markdown("### 🗂️ All as one JSON")
            st.caption("A single JSON array containing every document's result.")
            if st.button("Prepare Combined JSON", use_container_width=True):
                try:
                    combined = []
                    for doc in documents:
                        content = download_export(export_json_url(doc["document_id"]))
                        combined.append(json.loads(content))
                    combined_bytes = json.dumps(combined, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button(
                        "⬇️  Download All (JSON)",
                        data=combined_bytes,
                        file_name="all_documents.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Combined export failed: {exc}")

    with col_all_excel:
        with st.container(border=True):
            st.markdown("### 📊 All as one Excel")
            st.caption("One sheet with every document's rows stacked together.")
            if st.button("Prepare Combined Excel", use_container_width=True):
                try:
                    frames = []
                    for doc in documents:
                        content = download_export(export_excel_url(doc["document_id"]))
                        frames.append(pd.read_excel(io.BytesIO(content)))
                    combined_df = pd.concat(frames, ignore_index=True)

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        combined_df.to_excel(writer, index=False, sheet_name="all_documents")

                    st.download_button(
                        "⬇️  Download All (Excel)",
                        data=buffer.getvalue(),
                        file_name="all_documents.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Combined export failed: {exc}")
