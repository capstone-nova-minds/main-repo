"""Page 3: Human review and correction of extracted data (mandatory before export)."""

import pandas as pd
import streamlit as st

from utils.api_client import get_result, save_review
from utils.styling import apply_theme, render_sidebar_brand, render_header, render_stepper
from components.document_viewer import render_document_viewer
from components.extracted_fields_form import render_document_fields_form
from components.persons_table import render_persons_table

st.set_page_config(page_title="Review — Court Order Extraction", page_icon="🧾", layout="wide")


def render_measured_accuracy(evaluation: dict | None) -> None:
    """Measured Field Accuracy -- a *different* metric from the Extraction
    Quality Score on the Processing Status page: this one is a deterministic
    comparison of the automatic extraction against the human-reviewed
    values (ground truth), and only exists once a review has been saved.
    Never derived from OCR/NER confidence.
    """
    with st.container(border=True):
        st.markdown("#### 📏 Measured Field Accuracy")

        if evaluation is None:
            st.info("**Measured Accuracy: Pending Human Review**")
            st.caption(
                "This compares the automatic extraction against your saved review. "
                "Save a reviewed result above to calculate it."
            )
            return

        accuracy = evaluation.get("accuracy", 0.0)
        correct_fields = evaluation.get("correct_fields", 0)
        incorrect_fields = evaluation.get("incorrect_fields", 0)
        total_fields = evaluation.get("total_fields", 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy}%")
        col2.metric("Correct Fields", f"{correct_fields} / {total_fields}")
        col3.metric("Incorrect Fields", incorrect_fields)

        if accuracy >= 90:
            st.success("High measured extraction accuracy.")
        elif accuracy >= 75:
            st.warning("Moderate measured extraction accuracy.")
        else:
            st.error("Low measured extraction accuracy.")

        field_results = evaluation.get("field_results") or []

        with st.expander("🔍 View Accuracy Details"):
            if not field_results:
                st.caption("No field-level details available.")
            else:
                details_df = pd.DataFrame(field_results).rename(
                    columns={
                        "field": "Field",
                        "auto_value": "Automatic Value",
                        "reviewed_value": "Reviewed Value",
                        "correct": "Correct",
                    }
                )
                details_df = details_df[["Field", "Automatic Value", "Reviewed Value", "Correct"]]
                st.dataframe(details_df, use_container_width=True, hide_index=True)

apply_theme()
render_sidebar_brand()
render_header("🧾", "Review & Validation", "Check and correct the extracted fields before export. Human review is mandatory.")
render_stepper(3)

documents = st.session_state.get("documents", [])
if documents:
    doc_options = {f"{doc['filename']} ({doc['document_id'][:8]}…)": doc["document_id"] for doc in documents}
    labels = list(doc_options.keys())
    default_index = 0
    for i, doc_id in enumerate(doc_options.values()):
        if doc_id == st.session_state.get("document_id"):
            default_index = i
            break
    selected_label = st.selectbox("📄 Reviewing document:", labels, index=default_index)
    selected_id = doc_options[selected_label]
    if selected_id != st.session_state.get("document_id"):
        st.session_state["document_id"] = selected_id
        st.session_state.pop("extraction_result", None)
    st.write("")

if "document_id" not in st.session_state:
    st.warning("No document uploaded yet. Open **1 Upload** in the sidebar first.")
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

warnings = []

document_date = (result.get("document", {}).get("document_date") or {}).get("value")
if not document_date:
    warnings.append("لم يتم استخراج التاريخ -- يرجى إدخاله يدويًا إذا كان متوفرًا في المستند.")

if not result.get("persons"):
    warnings.append("لم يتم استخراج أي أشخاص -- يرجى إضافتهم يدويًا إذا لزم الأمر.")

if warnings:
    with st.container(border=True):
        st.markdown("##### ⚠️ Needs attention")
        for message in warnings:
            st.warning(message)

col_doc, col_view = st.columns([2, 1])

with col_doc:
    with st.container(border=True):
        st.markdown("#### 📋 Document Fields")
        edited_document = render_document_fields_form(result.get("document", {}), document_id)

    st.write("")
    with st.container(border=True):
        st.markdown("#### 👥 Persons / Companies")
        edited_persons = render_persons_table(result.get("persons", []), document_id)

with col_view:
    with st.container(border=True):
        st.markdown("#### 🖼️ Original Document")
        render_document_viewer(document_id)

# Cache this document's current edits every rerun (not just on Save click)
# so switching to another document and back doesn't lose them, and so
# "Save All" below can save every document that's been opened for review
# in this session -- not just whichever one is on screen right now.
st.session_state.setdefault("pending_reviews", {})
st.session_state["pending_reviews"][document_id] = {
    "document_id": document_id,
    "document": edited_document,
    "persons": edited_persons,
    "ocr_summary": result.get("ocr_summary", {}),
    "ner_summary": result.get("ner_summary", {}),
}

st.write("")

with st.container(border=True):
    st.markdown("#### 💾 Save")
    st.caption("Nothing is exported until you save your reviewed result here.")

    col_save_one, col_save_all = st.columns(2)

    with col_save_one:
        if st.button("💾  Save This Document", type="primary", use_container_width=True):
            payload = st.session_state["pending_reviews"][document_id]
            try:
                response = save_review(document_id, payload)
                st.session_state["reviewed_result"] = payload
                st.session_state["review_evaluation"] = response.get("evaluation")
                st.success("Reviewed result saved. Open **4 Export** in the sidebar to download the approved data.")
            except Exception as exc:
                st.error(f"Failed to save reviewed result: {exc}")

    with col_save_all:
        save_all_disabled = len(documents) <= 1
        if st.button(
            "💾  Save All Reviewed Documents",
            use_container_width=True,
            disabled=save_all_disabled,
        ):
            reviewed_ids = set(st.session_state["pending_reviews"].keys())
            not_yet_opened = [
                doc for doc in documents if doc["document_id"] not in reviewed_ids
            ]

            saved_count = 0
            errors = []
            for doc_id, payload in st.session_state["pending_reviews"].items():
                try:
                    save_review(doc_id, payload)
                    saved_count += 1
                except Exception as exc:
                    errors.append(f"{doc_id[:8]}…: {exc}")

            if saved_count:
                st.success(f"Saved {saved_count} reviewed document(s).")
            if not_yet_opened:
                names = ", ".join(doc["filename"] for doc in not_yet_opened)
                st.warning(
                    f"Skipped (never opened for review yet): {names}. "
                    "Select each one above at least once before Save All can include it."
                )
            if errors:
                st.error("Some documents failed to save:")
                for err in errors:
                    st.write(f"- {err}")

st.write("")
render_measured_accuracy(st.session_state.get("review_evaluation"))
