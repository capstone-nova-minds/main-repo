import streamlit as st
from config import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)

st.title("Tender Intelligence Assistant")

st.caption(
    "Login + Company Account + Persistent Knowledge Base + Tender History + RAG + Scoring"
)

st.success("Setup branch is working successfully.")

st.subheader("Project Workflow")

st.write(
    """
1. User logs in
2. User selects or creates company account
3. User fills or updates company profile
4. User uploads multiple tender files
5. System extracts tender text
6. System applies Arabic NLP preprocessing
7. System builds persistent knowledge base
8. System extracts key tender information
9. System compares tenders with company profile
10. System calculates score and ranking
11. User opens tender history
12. User asks questions using RAG chatbot
"""
)

st.subheader("Main Modules")

st.write(
    """
- Authentication and company account
- Tender file upload
- Arabic NLP preprocessing
- Structure-aware recursive chunking
- Persistent vector knowledge base
- Information extraction
- Tender scoring and ranking
- Tender history
- RAG chatbot
"""
)