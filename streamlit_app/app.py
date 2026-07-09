"""Landing page for the AI Court Order Extraction System (Streamlit)."""

import streamlit as st

st.set_page_config(page_title="AI Court Order Extraction System", page_icon="⚖️", layout="wide")

st.sidebar.title("AI Court Order Extraction System")
st.sidebar.info("Local OCR + Rule-based Extraction + Local Arabic NER | No LLM | No Cloud AI")

st.title("AI Court Order Extraction System")
st.write(
    "This tool extracts structured information from Arabic Jordanian court "
    "attachment orders, using only local OCR, rule-based extraction, and "
    "local Arabic NER. No LLM, no cloud AI, and no cloud OCR are used -- "
    "everything runs on this machine."
)

st.subheader("How to use this app")
st.markdown(
    """
    1. **Upload** — go to the **1_Upload** page and upload a PDF/JPG/JPEG/PNG court order.
    2. **Process** — go to **2_Processing_Status**, click *Start Processing*, and review OCR/NER status.
    3. **Review** — go to **3_Review_Validation** to check and correct the extracted fields and persons.
    4. **Export** — go to **4_Export** to download the approved data as JSON and Excel.

    Use the page navigation in the sidebar to move between steps.
    """
)

st.warning("Human review is mandatory before exporting any result.")
