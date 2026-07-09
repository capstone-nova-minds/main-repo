"""Central place for Arabic legal keywords and regex patterns.

Keeping these in one module makes it easy to tune extraction rules
without hunting through every service file.
"""

import re

# ---------------------------------------------------------------------------
# National ID
# ---------------------------------------------------------------------------
# Exactly 11 digits, not part of a longer digit run.
NATIONAL_ID_PATTERN = re.compile(r"(?<!\d)\d{11}(?!\d)")

# ---------------------------------------------------------------------------
# Document-level keywords
# ---------------------------------------------------------------------------
COURT_NAME_KEYWORDS = [
    "محكمة",
    "دائرة تنفيذ",
    "لدى محكمة",
    "محكمة بداية",
    "محكمة صلح",
]

CASE_NUMBER_KEYWORDS = [
    "رقم القضية",
    "رقم الدعوى",
    "رقم الملف",
    "القضية رقم",
    "دعوى رقم",
]

DOCUMENT_NUMBER_KEYWORDS = [
    "رقم الكتاب",
    "كتاب رقم",
    "رقم الصادر",
    "الرقم",
]

DOCUMENT_DATE_KEYWORDS = [
    "التاريخ",
    "تاريخ",
    "بتاريخ",
]

# Matches: 2026/07/09, 09/07/2026, 9-7-2026, etc. (digits already
# normalized to ASCII by normalization_service before this runs).
DATE_PATTERN = re.compile(r"\b\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}\b")

# ---------------------------------------------------------------------------
# Person / company keywords
# ---------------------------------------------------------------------------
PERSON_LINE_KEYWORDS = [
    "المطلوب",
    "المدين",
    "المحكوم عليه",
    "المحجوز عليه",
    "المذكورين",
    "السيد",
    "السادة",
    "شركة",
    "مؤسسة",
]

COMPANY_KEYWORDS = [
    "شركة",
    "مؤسسة",
    "سجل تجاري",
    "رقم تسجيل",
]

# ---------------------------------------------------------------------------
# Grouped name splitting
# ---------------------------------------------------------------------------
NAME_CONNECTORS = ["و", "،", "/"]

FAMILY_TAIL_KEYWORDS = [
    "أبناء",
    "اولاد",
    "أولاد",
    "ابن",
    "بن",
]
