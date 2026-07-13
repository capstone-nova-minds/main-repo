"""Landing page for the AI Court Order Extraction System (Streamlit)."""

import streamlit as st

from utils.styling import apply_theme, render_sidebar_brand, render_header, render_logo

st.set_page_config(page_title="AI Court Order Extraction System", page_icon="⚖️", layout="wide")

apply_theme()
render_sidebar_brand()

st.markdown(f'<div style="margin-bottom:1rem;">{render_logo(200, 58)}</div>', unsafe_allow_html=True)

render_header(
    "⚖️",
    "AI Court Order Extraction System",
    "Structured data from Arabic Jordanian court attachment orders — fully local, human-reviewed.",
)

trust_cols = st.columns(4)
trust_items = [
    ("🔒", "Fully Local", "Nothing leaves this machine"),
    ("🚫", "No LLM", "Rule-based + local NER only"),
    ("☁️", "No Cloud AI", "EasyOCR / Tesseract on-device"),
    ("👀", "Human Reviewed", "Review is mandatory before export"),
]
for col, (icon, title, desc) in zip(trust_cols, trust_items):
    with col:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)

st.write("")
st.subheader("How it works")

step_cols = st.columns(4)
step_items = [
    ("📤", "1. Upload", "Go to **Upload** and submit a PDF/JPG/JPEG/PNG court order."),
    ("⚙️", "2. Process", "Open **Processing Status**, click *Start Processing*, and check OCR/NER results."),
    ("🧾", "3. Review", "Open **Review & Validation** to correct fields and person records."),
    ("📦", "4. Export", "Open **Export** to download the approved data as JSON and Excel."),
]
for col, (icon, title, desc) in zip(step_cols, step_items):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

st.write("")
st.warning("⚠️ Human review is mandatory before exporting any result.")
st.caption("Use the page navigation in the sidebar to move between steps.")
