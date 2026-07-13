"""Editable table for person/company records, with review highlighting."""

import pandas as pd
import streamlit as st

PERSON_COLUMNS = [
    "full_name", "national_id", "registration_number",
    "person_type", "confidence", "needs_review", "source",
]

# Carried through the editor but never shown to the reviewer. record_index
# is the stable reference back to this row's original automatic record
# (stamped by api/process.py) -- evaluation_service.calculate_field_accuracy
# uses it to compare the right pair of records even after the reviewer
# edits full_name, national_id, or registration_number.
HIDDEN_COLUMNS = ["record_index"]

ALL_COLUMNS = PERSON_COLUMNS + HIDDEN_COLUMNS


def render_persons_table(persons: list) -> list:
    """Render an editable persons table, return the edited list of dicts."""
    if not persons:
        persons = [{
            "full_name": None, "national_id": None, "registration_number": None,
            "person_type": "Individual", "confidence": 0.0, "needs_review": True, "source": "rules",
        }]

    df = pd.DataFrame(persons)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[ALL_COLUMNS]

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
        column_order=PERSON_COLUMNS,  # record_index stays out of the UI
        column_config={
            "person_type": st.column_config.SelectboxColumn(options=["Individual", "Company"]),
            "confidence": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
            "needs_review": st.column_config.CheckboxColumn(),
        },
        key="persons_editor",
    )

    # A newly added row (num_rows="dynamic") has no record_index -- pandas
    # fills it (and any other blank cell) with NaN, which isn't valid JSON.
    # Converting to None also makes a missing record_index unambiguous:
    # this row has no corresponding original automatic record.
    edited_df = edited_df.where(pd.notnull(edited_df), None)
    records = edited_df.to_dict(orient="records")

    for record in records:
        record_index = record.get("record_index")
        if record_index is not None:
            record["record_index"] = int(record_index)

    return records
