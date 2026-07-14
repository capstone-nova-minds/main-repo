"""Shows the processed page images for a document, if available on disk.

The backend and frontend share the same data/ folder (a Docker volume in
containers, or the repo-root data/ folder when run locally), so we can
read page images directly instead of adding a dedicated API endpoint.
"""

import os
from pathlib import Path

import streamlit as st

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
PAGES_DIR = DATA_DIR / "pages"


@st.dialog("🔍 Original Document", width="large")
def _show_enlarged(image_path: str, caption: str) -> None:
    # Streamlit's dialog only offers "small"/"large" presets -- this CSS
    # override widens it further so the enlarged page is actually bigger,
    # not just the same size in a modal frame.
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 95vw !important;
            max-width: 1500px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image(image_path, caption=caption, use_container_width=True)


def render_document_viewer(document_id: str) -> None:
    page_dir = PAGES_DIR / document_id
    if not page_dir.exists():
        st.info("No page images found for this document yet.")
        return

    page_files = sorted(page_dir.glob("page_*.png"))
    if not page_files:
        st.info("No page images found for this document yet.")
        return

    for page_file in page_files:
        st.image(str(page_file), caption=page_file.stem, use_container_width=True)
        if st.button(
            "🔍 Click to enlarge",
            key=f"zoom_{document_id}_{page_file.stem}",
            use_container_width=True,
        ):
            _show_enlarged(str(page_file), page_file.stem)
