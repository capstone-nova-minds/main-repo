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
    "محكمة بداية",
    "محكمة صلح",
    "محكمة اربد",
    "دائرة تنفيذ",
    "لدى محكمة",
]

CASE_NUMBER_KEYWORDS = [

    "رقم القضية",
    "رقم الدعوى",
    "رقم الملف",
    "القضية رقم",
    "دعوى رقم",
    "بالدعوى رقم",
    "بدعوى رقم",
    "بالدعوى",
    "بدعوى",
]

DOCUMENT_NUMBER_KEYWORDS = [
    "رقم الكتاب",
    "كتاب رقم",
    "رقم الصادر",
    "الرقم",
    "رقم",

]

DOCUMENT_DATE_KEYWORDS = [
   "التاريخ",
    "تاريخ",
    "بتاريخ",
    "الموافق",
]

# Matches: 2026/07/09, 09/07/2026, 9-7-2026, etc. (digits already
# normalized to ASCII by normalization_service before this runs).
# Date formats:
# 2026/02/04
# 2026-02-04
# 04/02/2026
# 4-2-2026
DATE_PATTERN = re.compile(
    r"(?<!\d)("
    r"(?:20\d{2}|19\d{2})[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:0?[1-9]|[12]\d|3[01])"
    r"|"
    r"(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:20\d{2}|19\d{2})"
    r")(?!\d)"
)

# Capturing versions used to *normalize* a matched date to YYYY/MM/DD,
# regardless of which order/separator the OCR text used. Tried in this
# order by document_extraction_service._normalize_date_value:
#   1. Year-first, separator between every part: 2026/02/04, 2026-02-04
#   2. Day-first, separator between every part:   04/02/2026, 4-2-2026
#   3. Compact -- OCR sometimes drops separators: 20260204, 2026/0204
DATE_PATTERN_YMD = re.compile(
    r"(?<!\d)(20\d{2}|19\d{2})[\/\-.](0?[1-9]|1[0-2])[\/\-.](0?[1-9]|[12]\d|3[01])(?!\d)"
)
DATE_PATTERN_DMY = re.compile(
    r"(?<!\d)(0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|1[0-2])[\/\-.](20\d{2}|19\d{2})(?!\d)"
)
DATE_PATTERN_COMPACT = re.compile(
    r"(?<!\d)(20\d{2}|19\d{2})[\/\-.]?(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"
)
# ---------------------------------------------------------------------------
# Person / company keywords
# ---------------------------------------------------------------------------
NATIONAL_ID_PATTERN = re.compile(r"(?<!\d)\d{11}(?!\d)")

PERSON_LINE_KEYWORDS = [
    "للمستدعى ضده",
    "للمستدعى ضدها",
    "للمستدعى ضدهم",
    "المستدعى ضده",
    "المستدعى ضدها",
    "المستدعى ضدهم",
    "المطلوب ضده",
    "المطلوب ضدها",
    "المطلوب ضدهم",
    "المدين",
    "المحكوم عليه",
    "المحجوز عليه",
    "المذكورين",
    "السيد",
    "السادة",
]

COMPANY_KEYWORDS = [
    "شركة",
    "مؤسسة",
    "سجل تجاري",
    "رقم تسجيل",
    "رقم الشركة",
]

# Words that show up in property/legal-subject phrases and should never
# be accepted as part of a person's full_name (e.g. OCR/heuristic name
# extraction grabbing "قطعة الارض حوض ..." instead of an actual name).
INVALID_NAME_WORDS = {
    "الحجز",
    "التحفظي",
    "الموضوع",
    "قطعة",
    "الارض",
    "الأرض",
    "حوض",
    "لوحة",
    "قرية",
    "اراضي",
    "أراضي",
    "تسجيل",
    "مديرية",
    "حصصه",
    "حصته",
    "حصص",
    "بناء",
    "العائدة",
    "الاموال",
    "الأموال",
    "حقوق",
    "دعوى",
    "بداية",
    "القرار",
    "المبلغ",
    "تنفيذ",
    "حدود",
    "السيد",
    "مدير",
}

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


# Regex patterns
NATIONAL_ID_REGEX = r"(?<!\d)\d{11}(?!\d)"

# Supports: 2026/02/04, 2026-02-04, 04/02/2026, 4-2-2026
DATE_REGEXES = [
    r"(?<!\d)(20\d{2}|19\d{2})[\/\-.](0?[1-9]|1[0-2])[\/\-.](0?[1-9]|[12]\d|3[01])(?!\d)",
    r"(?<!\d)(0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|1[0-2])[\/\-.](20\d{2}|19\d{2})(?!\d)",
]

# Supports: 2026/41, 2026-41, and sometimes OCR gives 202641
CASE_NUMBER_REGEXES = [
    r"(?<!\d)(20\d{2}|19\d{2})\s*[\/\-]\s*(\d{1,6})(?!\d)",
    r"(?<!\d)(20\d{2}|19\d{2})(\d{1,4})(?!\d)",
]