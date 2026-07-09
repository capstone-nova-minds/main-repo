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
