"""Editable table for person/company records, with review highlighting."""

import pandas as pd
import streamlit as st

PERSON_COLUMNS = [
    "full_name", "national_id", "registration_number",
    "person_type", "confidence", "needs_review", "source",
]


def render_persons_table(persons: list) -> list:
    """Render an editable persons table, return the edited list of dicts."""
    if not persons:
        persons = [{
            "full_name": None, "national_id": None, "registration_number": None,
            "person_type": "Individual", "confidence": 0.0, "needs_review": True, "source": "rules",
        }]

    df = pd.DataFrame(persons)
    for col in PERSON_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[PERSON_COLUMNS]

    review_count = int(df["needs_review"].fillna(False).sum())
    if review_count:
        st.warning(f"{review_count} record(s) need review.")

    ner_only_no_id = df[(df["source"] == "ner") & (df["national_id"].isna())]
    if not ner_only_no_id.empty:
        st.warning("تم اكتشاف هذا الاسم بواسطة NER ويحتاج مراجعة.")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "person_type": st.column_config.SelectboxColumn(options=["Individual", "Company"]),
            "confidence": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
            "needs_review": st.column_config.CheckboxColumn(),
        },
        key="persons_editor",
    )

    return edited_df.to_dict(orient="records")
