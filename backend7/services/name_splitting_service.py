"""Split grouped family names into individual person records.

Example:
  "أحمد ومحمد وخالد أبناء محمود سالم"
  ->
  [أحمد محمود سالم, محمد محمود سالم, خالد محمود سالم]

If the pattern is not clearly "first-names + family tail", we leave the
name unsplit and flag it for human review rather than guessing.
"""

import re
from typing import List

from schemas.person_schema import PersonRecord
from utils.regex_patterns import FAMILY_TAIL_KEYWORDS

# Split first-name lists on و / ، / for grouped-name detection.
_CONNECTOR_SPLIT = re.compile(r"\s*(?:و|،|/)\s*")


def _find_family_tail(name: str):
    """Return (first_names_part, family_part) if a family-tail keyword is found."""
    for keyword in FAMILY_TAIL_KEYWORDS:
        pattern = re.compile(rf"\s{re.escape(keyword)}\s")
        match = pattern.search(f" {name} ")
        if match:
            idx = name.find(keyword)
            if idx > 0:
                first_names_part = name[:idx].strip()
                family_part = name[idx + len(keyword):].strip()
                return first_names_part, family_part
    return None, None


def split_grouped_name(name: str) -> List[str]:
    """Split one grouped-name string into a list of full names.

    Returns a single-item list [name] unchanged if no clear grouping is
    detected (caller should keep needs_review=true in that case).
    """
    if not name:
        return [name]

    first_names_part, family_part = _find_family_tail(name)
    if not first_names_part or not family_part:
        return [name]

    first_names = [n.strip() for n in _CONNECTOR_SPLIT.split(first_names_part) if n.strip()]
    if len(first_names) < 2:
        # Not actually a group of names -- nothing to split.
        return [name]

    return [f"{first_name} {family_part}".strip() for first_name in first_names]


def expand_grouped_person_records(records: List[PersonRecord]) -> List[PersonRecord]:
    """For each Individual record, split grouped names into separate records."""
    expanded: List[PersonRecord] = []

    for record in records:
        if record.person_type != "Individual" or not record.full_name:
            expanded.append(record)
            continue

        split_names = split_grouped_name(record.full_name)

        if len(split_names) == 1:
            expanded.append(record)
            continue

        # A successful split can't keep a single shared National ID across
        # multiple people -- each new record needs review since we can't
        # tell which split name (if any) the ID actually belongs to.
        for split_name in split_names:
            expanded.append(PersonRecord(
                full_name=split_name,
                national_id=None,
                registration_number=record.registration_number,
                person_type="Individual",
                confidence=record.confidence,
                needs_review=True,
                source=record.source,
            ))

    return expanded
