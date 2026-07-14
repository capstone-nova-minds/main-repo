"""Rule-based extraction of person/company candidate records.

This service extracts persons and companies from normalized OCR text using
five methods, from most to least precise:

0. Structured target-list extraction.
1. Direct legal-phrase + name + National ID regex.
2. Legal-keyword candidate lines.
3. Line-context extraction around every valid National ID.
4. Emergency wide-context fallback around National ID.
5. OCR orphan first-name recovery.

No LLM and no cloud AI are used.
"""

import re
from typing import List, Optional, Tuple

from schemas.person_schema import PersonRecord
from utils.confidence import (
    EXACT_MATCH_CONFIDENCE,
    NEARBY_MATCH_CONFIDENCE,
    WEAK_MATCH_CONFIDENCE,
)
from utils.regex_patterns import (
    NATIONAL_ID_PATTERN,
    PERSON_LINE_KEYWORDS,
    COMPANY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEADING_NOISE = re.compile(r"^[:\-\s،]+")
_MULTI_SPACE = re.compile(r"\s+")
_ARABIC_WORD_PATTERN = re.compile(r"[\u0600-\u06FF]{2,}")

MIN_NAME_WORDS = 3


_STRUCTURED_LABEL_NOISE = {
    "نوع",
    "الشخص",
    "فرد",
    "شركة",
    "الرقم",
    "المطلوبة",
    "المطلوبه",
    "المطلوية",
    "الأشخاص",
    "الاشخاص",
    "الجهات",
    "الوطني",
    "رقم",
    "وطني",
    "التسجيل",
    "الأشخاص",
    "الاشخاص",
    "الجهات",
    "المطلوبة",
    "المطلوبه",
    "المعلومات",
    "المتوفرة",
    "الواردة",
    "أدناه",
    "ادناه",
    "حول",
    "الأسماء",
    "الاسماء",
    "قرار",
    "المحكمة",
    "يرجى",
    "اتخاذ",
    "الاجراءات",
    "الإجراءات",
    "حسب",
    "الأصول",
    "الاصول",
    "استنادا",
    "استنادأ",
    "لديكم",
    "وتزويدنا",
    "تاريخ",
    "امر",
    "أمر",
    "اللازمة",
}

INVALID_NAME_WORDS = {
    "الحجز",
    "التحفظي",
    "الموضوع",
    "قطعة",
    "الارض",
    "الأرض",
    "ارض",
    "أرض",
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
    "المحترم",
    "نوع",
    "الشخص",
    "فرد",
    "الرقم",
    "الوطني",
    "إلى",
    "الى",
    "السادة",
    "تحية",
    "طيبة",
    "طبية",
    "وبعد",
    "المعلومات",
    "المتوفرة",
    "الأشخاص",
    "الاشخاص",
    "الجهات",
    "المطلوبة",
    "الواردة",
    "أدناه",
    "ادناه",
}


_NAME_NOISE_WORDS = {
    "رقم",
    "وطني",
    "دعوى",
    "بداية",
    "حقوق",
    "اربد",
    "إربد",
    "لقد",
    "تقرر",
    "بالدعوى",
    "بدعوى",
    "ضده",
    "ضدا",
    "ضدها",
    "ضدهم",
    "المستدعى",
    "المستدعي",
    "على",
    "الاموال",
    "الأموال",
    "العائدة",
    "للمستدعى",
    "للمستدعي",
    "ومن",
    "ضمنها",
    "ضمنهم",
}


_ID_STOP_PHRASES = [
    "و رقم وطني",
    "ورقم وطني",
    "رقم وطني",
    "رقمه الوطني",
    "الرقم الوطني",
    "حامل الرقم الوطني",
    "ومن ضمنها",
    "ومن ضمنهم",
    "وحصته",
    "و حصته",
    "وحصصه",
    "و حصصه",
    "وذلك",
    "بحدود",
    "تنفيذا",
    "لتنفيذ",
]


_PERSON_HINTS = [
    "للمستدعى ضده",
    "للمستدعي ضده",
    "المستدعى ضده",
    "المستدعي ضده",
    "المستدعى ضدها",
    "المستدعى ضدهم",
    "المطلوب ضده",
    "المطلوب ضدها",
    "المطلوب ضدهم",
    "المدين",
    "المحكوم عليه",
    "المحجوز عليه",
    "السيد",
]


_DIRECT_PHRASE_KEYWORDS = [
    "للمستدعى ضده",
    "للمستدعي ضده",
    "المستدعى ضده",
    "المستدعي ضده",
    "المستدعى ضدها",
    "المستدعى ضدهم",
    "المطلوب ضده",
    "المطلوب ضدها",
    "المطلوب ضدهم",
    "المدين",
    "المحكوم عليه",
    "المحجوز عليه",
]


_ID_CONNECTOR = (
    r"(?:"
    r"و\s*رقم\s*وطني|"
    r"ورقم\s*وطني|"
    r"رقم\s*وطني|"
    r"الرقم\s*الوطني|"
    r"رقمه\s*الوطني"
    r")"
)


_DIRECT_PHRASE_PATTERNS = [
    re.compile(
        re.escape(phrase)
        + r"\s*[:\-]?\s*"
        + r"(?P<name>[\u0600-\u06FF\s]{5,120}?)\s*"
        + _ID_CONNECTOR
        + r"\s*[:\-]?\s*[\(\[]?\s*"
        + r"(?P<id>\d{11})"
        + r"\s*[\)\]]?"
    )
    for phrase in _DIRECT_PHRASE_KEYWORDS
]


_OCR_NAME_FIXES = {
    "بوسف": "يوسف",
    "المصرى": "المصري",
    "احمد": "أحمد",
    "عىد": "عبد",
    "عدد": "عبد",
    "عبدد": "عبد",
    "عىبد": "عبد",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _record_kwargs(**kwargs):
    """Only pass fields that exist in PersonRecord."""
    try:
        fields = set(PersonRecord.model_fields.keys())
    except AttributeError:
        fields = set(PersonRecord.__fields__.keys())

    return {key: value for key, value in kwargs.items() if key in fields}


def _make_person_record(**kwargs) -> PersonRecord:
    return PersonRecord(**_record_kwargs(**kwargs))


def _copy_record_with_updates(record: PersonRecord, **updates) -> PersonRecord:
    """Return updated PersonRecord without mutating the original."""
    if hasattr(record, "model_dump"):
        data = record.model_dump()
    else:
        data = record.dict()

    data.update(updates)
    return PersonRecord(**data)


def _normalize_spaces(text: str) -> str:
    if not text:
        return ""
    return _MULTI_SPACE.sub(" ", text).strip()


def _fix_ocr_name_word(word: str) -> str:
    return _OCR_NAME_FIXES.get(word, word)


def _clean_words(words: List[str]) -> List[str]:
    cleaned = []

    for word in words:
        word = _fix_ocr_name_word(word)

        if word in _NAME_NOISE_WORDS:
            continue

        cleaned.append(word)

    return cleaned


def _cut_before_stop_words(text: str) -> str:
    if not text:
        return ""

    for stop_word in _ID_STOP_PHRASES:
        if stop_word in text:
            text = text.split(stop_word, 1)[0]

    return text


def is_company_line(line: str) -> bool:
    if not line:
        return False
    return any(keyword in line for keyword in COMPANY_KEYWORDS)


def find_national_ids(text: str) -> List[str]:
    if not text:
        return []
    return NATIONAL_ID_PATTERN.findall(text)


def is_valid_person_name(
    name: Optional[str],
    has_national_id: bool = False,
    direct_pattern_match: bool = False,
) -> bool:
    """Reject false positives like headers/legal words."""
    if not name:
        return False

    words = name.split()

    if any(word in _STRUCTURED_LABEL_NOISE for word in words):
        return False

    if any(word in INVALID_NAME_WORDS for word in words):
        return False

    if len(words) < MIN_NAME_WORDS and not (has_national_id and direct_pattern_match):
        return False

    return True


def extract_candidate_lines(cleaned_text: str) -> List[str]:
    """Return lines that mention a person/company legal keyword."""
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    candidates = []

    for i, line in enumerate(lines):
        if any(keyword in line for keyword in PERSON_LINE_KEYWORDS):
            candidate = line

            if i + 1 < len(lines):
                candidate = f"{candidate} {lines[i + 1]}"

            candidates.append(candidate)

    return candidates


def _matched_keyword(line: str) -> Optional[str]:
    for keyword in sorted(PERSON_LINE_KEYWORDS, key=len, reverse=True):
        if keyword in line:
            return keyword
    return None


def _normalize_registration_number(value: str) -> Optional[str]:
    """Normalize company registration number like REG - 202602 -> REG-202602.

    Also recovers the fixed "REG" prefix from the common OCR misread
    "RE6" (the letter G is visually confused with the digit 6 in some
    fonts) -- the same normalization-recovery approach already used for
    the "UW" document-number prefix.
    """
    if not value:
        return None

    value = str(value).strip()
    value = re.sub(r"\s*[-–—]\s*", "-", value)
    value = re.sub(r"\s+", "", value)
    value = value.upper()

    # Recover "REG" from the common "RE6" misread (G <-> 6 confusion).
    value = re.sub(r"^RE6(?=-)", "REG", value)

    if re.fullmatch(r"[A-Z]{1,10}-\d{3,15}", value):
        return value

    if re.fullmatch(r"\d{3,15}", value):
        return value


def _extract_registration_number(line: str) -> Optional[str]:
    """
    Extract company registration number.

    Supports:
    - رقم التسجيل: 123456
    - رقم التسجيل: REG-202602
    - REG-202602 : رقم التسجيل
    """
    if not line:
        return None

    patterns = [
        r"(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)\s*[:：]?\s*(?P<value>[A-Za-z]{1,10}\s*[-–—]\s*\d{3,15}|\d{3,15})",
        r"(?P<value>[A-Za-z]{1,10}\s*[-–—]\s*\d{3,15}|\d{3,15})\s*[:：]?\s*(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)",
    ]

    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)

        if match:
            return _normalize_registration_number(match.group("value"))

    return None

# ---------------------------------------------------------------------------
# Method 0: Structured target list extraction
# ---------------------------------------------------------------------------

_TARGET_SECTION_KEYWORDS = [
    "الأشخاص / الجهات المطلوبة",
    "الاشخاص / الجهات المطلوبة",
    "الأشخاص/ الجهات المطلوبة",
    "الاشخاص/ الجهات المطلوبة",
    "الأشخاص الجهات المطلوبة",
    "الاشخاص الجهات المطلوبة",
    "الأشخاص المطلوبة",
    "الاشخاص المطلوبة",
    "الجهات المطلوبة",
    "الجهات المطلوبه",
]


_RECIPIENT_OR_GREETING_PHRASES = [
    "إلى السادة",
    "الى السادة",
    "السادة",
    "شركة المحفظة الالكترونية",
    "شركة المحفظة الإلكترونية",
    "تحية طيبة",
    "تحية طبية",
    "وبعد",
]


_SECTION_END_KEYWORDS = [
    "مرفق",
    "وتفضلوا",
    "ختم",
    "كاتب",
    "صفحة",
    "يرجى اعتماد",
    "هذه الصفحة",
]


def _is_recipient_or_greeting_line(line: str) -> bool:
    """Reject recipient/greeting lines that are not target persons."""
    if not line:
        return False

    if find_national_ids(line):
        return False

    if _extract_registration_number(line):
        return False

    return any(phrase in line for phrase in _RECIPIENT_OR_GREETING_PHRASES)


def _is_section_end(line: str) -> bool:
    if not line:
        return False

    return any(keyword in line for keyword in _SECTION_END_KEYWORDS)


def _get_target_section_lines(cleaned_text: str) -> List[str]:
    """
    Return only the target persons/companies section.

    Returns [] (not the whole document) if no heading trigger is found.
    This method is only safe to use on documents that actually have this
    structured template -- silently falling back to the whole text
    previously caused it to grab header/footer garbage on old-style
    documents that don't use this heading at all.

    Triggers on the single word "الأشخاص" rather than requiring the full
    "الأشخاص / الجهات المطلوبة" phrase to match exactly: OCR frequently
    misreads "المطلوبة" (e.g. as "المطلوية") or splits the heading across
    several lines, so requiring the full phrase caused this method to
    never fire at all on otherwise-correctly-ordered text.
    """
    if not cleaned_text:
        return []

    trigger_pos = None

    for keyword in ("الأشخاص", "الاشخاص"):
        pos = cleaned_text.find(keyword)
        if pos != -1 and (trigger_pos is None or pos < trigger_pos):
            trigger_pos = pos

    if trigger_pos is None:
        return []

    search_text = cleaned_text[trigger_pos:]

    lines = [line.strip() for line in search_text.splitlines() if line.strip()]

    target_lines = []

    for line in lines:
        if _is_section_end(line):
            break

        target_lines.append(line)

    return target_lines


def _is_noise_word(word: str, allow_company_word: bool = False) -> bool:
    if allow_company_word and word == "شركة":
        return False

    return (
        word in _STRUCTURED_LABEL_NOISE
        or word in INVALID_NAME_WORDS
        or word in _NAME_NOISE_WORDS
    )


def _extract_name_segments(
    text: str,
    allow_company_word: bool = False,
) -> List[List[str]]:
    """
    Split Arabic words into possible name segments.
    Noise words break the segment.
    """
    if not text:
        return []

    text = text.replace("|", " ")
    text = text.replace(":", " ")
    text = text.replace("：", " ")
    text = text.replace("-", " ")
    text = text.replace("،", " ")
    text = NATIONAL_ID_PATTERN.sub(" ", text)
    text = re.sub(r"[A-Za-z]{1,10}\s*[-–—]\s*\d{3,15}", " ", text)
    text = _normalize_spaces(text)

    words = _ARABIC_WORD_PATTERN.findall(text)
    words = [_fix_ocr_name_word(word) for word in words]

    segments: List[List[str]] = []
    current: List[str] = []

    for word in words:
        if _is_noise_word(word, allow_company_word=allow_company_word):
            if current:
                segments.append(current)
                current = []
            continue

        current.append(word)

    if current:
        segments.append(current)

    return segments


def _clean_structured_person_name(raw_name: str) -> Optional[str]:
    """Clean a person name captured from structured target-list rows."""
    segments = _extract_name_segments(raw_name, allow_company_word=False)

    valid_segments = []

    for segment in segments:
        if len(segment) >= 3:
            name = _normalize_spaces(" ".join(segment[-6:]))

            if is_valid_person_name(name, has_national_id=True):
                valid_segments.append(name)

    if not valid_segments:
        return None

    return valid_segments[-1]


def _clean_structured_company_name(raw_name: str) -> Optional[str]:
    """Clean company name from structured target-list rows."""
    segments = _extract_name_segments(raw_name, allow_company_word=True)

    candidates = []

    for segment in segments:
        if len(segment) < 2:
            continue

        name = _normalize_spaces(" ".join(segment[-8:]))

        if any(keyword in name for keyword in COMPANY_KEYWORDS):
            candidates.append(name)

    if not candidates:
        return None

    return max(candidates, key=lambda item: len(item.split()))


def _extract_name_near_national_id(
    section_text: str,
    national_id: str,
    search_from: int = 0,
) -> "tuple[Optional[str], int]":
    """
    Extract individual name around National ID from flattened target section.

    Handles OCR order issues:
    - name before ID
    - name after ID
    - fields split into different lines

    Returns (name, id_pos). id_pos is -1 if this national_id doesn't occur
    at or after search_from. Callers use id_pos to keep searching past this
    occurrence -- the same ID text can appear more than once (e.g. an OCR
    misread duplicating a digit between two different people), and each
    occurrence should get its own chance at a name instead of every
    occurrence but the first being silently skipped.
    """
    if not section_text or not national_id:
        return None, -1

    id_pos = section_text.find(national_id, search_from)

    if id_pos == -1:
        return None, -1

    before_id = section_text[max(0, id_pos - 220):id_pos]
    after_id = section_text[id_pos + len(national_id): id_pos + len(national_id) + 180]

    # Usually Arabic form has name before "الرقم الوطني".
    before_name = _clean_structured_person_name(before_id)

    if before_name:
        return before_name, id_pos

    after_name = _clean_structured_person_name(after_id)

    if after_name:
        return after_name, id_pos

    return None, id_pos


def _find_registration_numbers(text: str) -> List[str]:
    if not text:
        return []

    patterns = [
        r"(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)\s*[:：]?\s*(?P<value>[A-Za-z0-9]{1,10}\s*[-–—]\s*\d{3,15}|\d{3,15})",
        r"(?P<value>[A-Za-z0-9]{1,10}\s*[-–—]\s*\d{3,15}|\d{3,15})\s*[:：]?\s*(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)",
        r"(?P<value>RE[G6]\s*[-–—]\s*\d{3,15})",
    ]

    values = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _normalize_registration_number(match.group("value"))

            if value and value not in values:
                values.append(value)

    return values


def _extract_company_name_near_registration(
    section_text: str,
    registration_number: str,
) -> Optional[str]:
    """
    Extract company name around registration number.

    Example:
    شركة النجمة للحلول التقنية | رقم التسجيل: REG-202602 | نوع الشخص: شركة
    """
    if not section_text or not registration_number:
        return None

    digits_match = re.search(r"(\d{3,15})$", registration_number)
    reg_pos = -1
    match_len = len(registration_number)

    if digits_match:
        digits = digits_match.group(1)
        loose_match = re.search(
            r"[A-Za-z0-9]{1,10}\s*[-–—]\s*" + re.escape(digits),
            section_text,
        )

        if loose_match:
            reg_pos = loose_match.start()
            match_len = loose_match.end() - loose_match.start()

    if reg_pos == -1:
        reg_pos = section_text.find(registration_number)

        if reg_pos == -1:
            return None

    before_reg = section_text[max(0, reg_pos - 220):reg_pos]
    after_reg = section_text[reg_pos + match_len: reg_pos + match_len + 120]

    before_company = _clean_structured_company_name(before_reg)

    if before_company:
        return before_company

    after_company = _clean_structured_company_name(after_reg)

    if after_company:
        return after_company


def _extract_structured_list_records(cleaned_text: str) -> List[PersonRecord]:
    """
    Extract target rows from:
    الأشخاص / الجهات المطلوبة:
    - name | الرقم الوطني: id | نوع الشخص: فرد
    - company name | رقم التسجيل: REG-202602 | نوع الشخص: شركة
    """
    if not cleaned_text:
        return []

    records: List[PersonRecord] = []

    target_lines = _get_target_section_lines(cleaned_text)
    section_text = _normalize_spaces(" ".join(target_lines))

    if not section_text:
        return []

    # ------------------------------------------------------------
    # Individuals: National ID-based extraction.
    #
    # Deliberately does NOT dedupe the ID list before extracting names:
    # the same ID text can legitimately appear more than once (e.g. an
    # OCR misread duplicating a digit between two different siblings), and
    # collapsing to unique IDs first meant every occurrence past the first
    # was silently skipped, dropping a real person from the result.
    # Instead we walk every occurrence by position; _deduplicate_records
    # below still merges true duplicates (same ID *and* same name).
    # ------------------------------------------------------------
    search_from = 0

    for national_id in find_national_ids(section_text):
        full_name, id_pos = _extract_name_near_national_id(
            section_text, national_id, search_from=search_from
        )

        if id_pos == -1:
            continue

        search_from = id_pos + len(national_id)

        if not full_name:
            continue

        records.append(
            _make_person_record(
                full_name=full_name,
                national_id=national_id,
                registration_number=None,
                person_type="Individual",
                confidence=0.90,
                needs_review=False,
                source="rules",
                extraction_method="structured_target_list",
            )
        )

    # ------------------------------------------------------------
    # Companies: Registration Number-based extraction.
    # ------------------------------------------------------------
    registration_numbers = _find_registration_numbers(section_text)

    for registration_number in registration_numbers:
        company_name = _extract_company_name_near_registration(
            section_text,
            registration_number,
        )

        if not company_name:
            continue

        records.append(
            _make_person_record(
                full_name=company_name,
                national_id=None,
                registration_number=registration_number,
                person_type="Company",
                confidence=0.90,
                needs_review=False,
                source="rules",
                extraction_method="structured_target_list",
            )
        )

    return _deduplicate_records(records)
# ---------------------------------------------------------------------------
# Method 1: Direct legal phrase + name + National ID
# ---------------------------------------------------------------------------

def _clean_direct_match_name(raw_name: str) -> Optional[str]:
    text = _cut_before_stop_words(raw_name)
    words = _ARABIC_WORD_PATTERN.findall(text)
    words = _clean_words(words)

    if not words:
        return None

    name = _normalize_spaces(" ".join(words))

    if not is_valid_person_name(
        name,
        has_national_id=True,
        direct_pattern_match=True,
    ):
        return None

    return name


def _extract_direct_legal_phrase_records(cleaned_text: str) -> List[PersonRecord]:
    if not cleaned_text:
        return []

    search_text = re.sub(r"\s+", " ", cleaned_text)
    records: List[PersonRecord] = []

    # No same-ID skip here: two different people can legitimately share an
    # (possibly OCR-misread) ID, and the caller's final _deduplicate_records
    # pass already handles same-ID collisions correctly -- keeping both and
    # flagging for review when the names differ, merging true duplicates
    # when they match.
    for pattern in _DIRECT_PHRASE_PATTERNS:
        for match in pattern.finditer(search_text):
            national_id = match.group("id")
            name = _clean_direct_match_name(match.group("name"))

            if not name:
                continue

            records.append(
                _make_person_record(
                    full_name=name,
                    national_id=national_id,
                    registration_number=None,
                    person_type="Individual",
                    confidence=float(EXACT_MATCH_CONFIDENCE),
                    needs_review=False,
                    source="rules",
                    extraction_method="direct_legal_phrase",
                )
            )

    return records


# ---------------------------------------------------------------------------
# Method 2: Keyword candidate line
# ---------------------------------------------------------------------------

def _clean_name(line: str, keyword: str) -> Optional[str]:
    if not line or not keyword:
        return None

    if keyword in line:
        name_part = line.split(keyword, 1)[1]
    else:
        name_part = line

    id_match = NATIONAL_ID_PATTERN.search(name_part)

    if id_match:
        name_part = name_part[:id_match.start()]

    registration_number = _extract_registration_number(name_part)

    if registration_number:
        name_part = name_part.replace(registration_number, " ")

    name_part = _cut_before_stop_words(name_part)
    name_part = NATIONAL_ID_PATTERN.sub(" ", name_part)
    name_part = name_part.replace("(", " ").replace(")", " ")
    name_part = name_part.replace("[", " ").replace("]", " ")
    name_part = name_part.replace(":", " ").replace("：", " ")
    name_part = name_part.strip(" :-،,؛.")
    name_part = _LEADING_NOISE.sub("", name_part)
    name_part = _normalize_spaces(name_part)

    words = _ARABIC_WORD_PATTERN.findall(name_part)
    words = _clean_words(words)

    name = _normalize_spaces(" ".join(words))

    if len(name) < 5:
        return None

    return name


def _build_record_from_line(line: str) -> Optional[PersonRecord]:
    if _is_recipient_or_greeting_line(line):
        return None

    keyword = _matched_keyword(line)

    if keyword is None:
        return None

    national_ids = find_national_ids(line)
    national_id = national_ids[0] if national_ids else None

    full_name = _clean_name(line, keyword)

    if not full_name:
        return None

    is_company = is_company_line(line) or is_company_line(full_name)
    registration_number = _extract_registration_number(line)

    if is_company:
        return _make_person_record(
            full_name=full_name,
            national_id=None,
            registration_number=registration_number,
            person_type="Company",
            confidence=float(NEARBY_MATCH_CONFIDENCE),
            needs_review=True,
            source="rules",
            extraction_method="keyword_line",
        )

    if not is_valid_person_name(full_name, has_national_id=bool(national_id)):
        return None

    if national_id:
        return _make_person_record(
            full_name=full_name,
            national_id=national_id,
            registration_number=None,
            person_type="Individual",
            confidence=float(EXACT_MATCH_CONFIDENCE),
            needs_review=False,
            source="rules",
            extraction_method="keyword_line",
        )

    return _make_person_record(
        full_name=full_name,
        national_id=None,
        registration_number=None,
        person_type="Individual",
        confidence=float(WEAK_MATCH_CONFIDENCE),
        needs_review=True,
        source="rules",
        extraction_method="keyword_line",
    )


# ---------------------------------------------------------------------------
# Method 3: Context around National ID
# ---------------------------------------------------------------------------

def _extract_name_from_id_line_context(
    lines: List[str],
    line_index: int,
) -> Optional[Tuple[str, str]]:
    line = lines[line_index]
    ids_on_line = find_national_ids(line)

    if not ids_on_line:
        return None

    national_id = ids_on_line[0]

    prev_line = lines[line_index - 1] if line_index > 0 else ""
    next_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""

    context = f"{line} {next_line} {prev_line}"
    context = context.replace("(", " ").replace(")", " ")
    context = context.replace("،", " ")
    context = _normalize_spaces(context)

    id_pos = context.find(national_id)

    if id_pos == -1:
        return None

    before_id = context[:id_pos].strip()
    after_id = context[id_pos + len(national_id):].strip()

    name_candidates = [before_id, after_id]

    for candidate_text in name_candidates:
        name_part = candidate_text

        for hint in _PERSON_HINTS:
            if hint in name_part:
                name_part = name_part.split(hint, 1)[1]
                break

        name_part = _cut_before_stop_words(name_part)

        words = _ARABIC_WORD_PATTERN.findall(name_part)
        words = _clean_words(words)

        if len(words) < MIN_NAME_WORDS:
            continue

        words = words[-5:]
        name = _normalize_spaces(" ".join(words))

        if is_valid_person_name(name, has_national_id=True):
            return name, national_id

    return None


def _extract_context_records(cleaned_text: str, skip_ids: set) -> List[PersonRecord]:
    if not cleaned_text:
        return []

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    records: List[PersonRecord] = []

    for i, line in enumerate(lines):
        if not find_national_ids(line):
            continue

        result = _extract_name_from_id_line_context(lines, i)

        if result is None:
            continue

        name, national_id = result

        if national_id in skip_ids:
            continue

        records.append(
            _make_person_record(
                full_name=name,
                national_id=national_id,
                registration_number=None,
                person_type="Individual",
                confidence=float(EXACT_MATCH_CONFIDENCE),
                needs_review=False,
                source="rules",
                extraction_method="national_id_context",
            )
        )

    return records


# ---------------------------------------------------------------------------
# Method 4: Emergency fallback around National ID
# ---------------------------------------------------------------------------

def _is_valid_emergency_person_name(name: Optional[str]) -> bool:
    if not name:
        return False

    words = name.split()

    if len(words) < 2:
        return False

    if any(word in INVALID_NAME_WORDS for word in words):
        return False

    if any(word in _STRUCTURED_LABEL_NOISE for word in words):
        return False

    if all(word in _NAME_NOISE_WORDS for word in words):
        return False

    return True


def _extract_emergency_national_id_records(
    cleaned_text: str,
    skip_ids: set,
) -> List[PersonRecord]:
    if not cleaned_text:
        return []

    text = re.sub(r"\s+", " ", cleaned_text).strip()
    national_ids = find_national_ids(text)

    records: List[PersonRecord] = []

    for national_id in national_ids:
        if national_id in skip_ids:
            continue

        id_pos = text.find(national_id)

        if id_pos == -1:
            continue

        before_id = text[max(0, id_pos - 450):id_pos]
        after_id = text[id_pos + len(national_id): id_pos + len(national_id) + 180]

        candidate_texts = [after_id, before_id]

        for candidate_text in candidate_texts:
            candidate_text = _cut_before_stop_words(candidate_text)
            candidate_text = candidate_text.replace("(", " ")
            candidate_text = candidate_text.replace(")", " ")
            candidate_text = candidate_text.replace("[", " ")
            candidate_text = candidate_text.replace("]", " ")
            candidate_text = candidate_text.replace("،", " ")
            candidate_text = candidate_text.replace(":", " ")
            candidate_text = candidate_text.replace("-", " ")
            candidate_text = _normalize_spaces(candidate_text)

            words = _ARABIC_WORD_PATTERN.findall(candidate_text)
            words = _clean_words(words)

            if not words:
                continue

            segments = []
            current_segment = []

            for word in words:
                if word in INVALID_NAME_WORDS or word in _STRUCTURED_LABEL_NOISE:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
                    continue

                current_segment.append(word)

            if current_segment:
                segments.append(current_segment)

            valid_segments = [
                segment
                for segment in segments
                if len(segment) >= 3
            ]

            if not valid_segments:
                continue

            selected_words = valid_segments[0] if candidate_text == after_id else valid_segments[-1]
            selected_words = selected_words[-5:]

            full_name = _normalize_spaces(" ".join(selected_words))

            if not _is_valid_emergency_person_name(full_name):
                continue

            records.append(
                _make_person_record(
                    full_name=full_name,
                    national_id=national_id,
                    registration_number=None,
                    person_type="Individual",
                    confidence=0.65,
                    needs_review=True,
                    source="rules",
                    extraction_method="emergency_national_id_context",
                )
            )
            break

    return records


# ---------------------------------------------------------------------------
# OCR orphan first-name recovery
# ---------------------------------------------------------------------------

_COMMON_FIRST_NAMES = {
    "عمر",
    "محمد",
    "احمد",
    "أحمد",
    "خالد",
    "نور",
    "هدى",
    "هدا",
    "ابراهيم",
    "إبراهيم",
    "سامي",
    "ناصر",
    "مازن",
    "محمود",
    "عبدالله",
    "عبد",
    "سارة",
    "رنا",
}


_ORPHAN_NOISE_WORDS = {
    "الرقم",
    "الوطني",
    "فرد",
    "نوع",
    "الشخص",
    "ختم",
    "كاتب",
    "المحكمة",
    "صفحة",
    "أمر",
    "امر",
    "أموال",
    "اموال",
    "على",
    "اللازمة",
    "وتزويدنا",
    "افي",
    "في",
}


def _extract_orphan_first_names(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return []

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    candidates: List[str] = []

    start_index = 0

    for i, line in enumerate(lines):
        if "الأشخاص" in line or "الاشخاص" in line or "الجهات المطلوبة" in line:
            start_index = i
            break

    for line in lines[start_index:]:
        words = _ARABIC_WORD_PATTERN.findall(line)

        if len(words) != 1:
            continue

        word = _fix_ocr_name_word(words[0])

        if word in _ORPHAN_NOISE_WORDS:
            continue

        if word in _COMMON_FIRST_NAMES:
            candidates.append(word)

    unique = []

    for word in candidates:
        if word not in unique:
            unique.append(word)

    return unique


def _recover_orphan_first_names(
    records: List[PersonRecord],
    cleaned_text: str,
) -> List[PersonRecord]:
    orphan_first_names = _extract_orphan_first_names(cleaned_text)

    if not orphan_first_names:
        return records

    updated_records: List[PersonRecord] = []
    orphan_index = 0

    for record in records:
        full_name = record.full_name or ""
        words = full_name.split()

        should_try_recovery = (
            record.person_type == "Individual"
            and bool(record.national_id)
            and len(words) == 3
            and getattr(record, "extraction_method", None) in {
                "structured_target_list",
                "national_id_context",
                "emergency_national_id_context",
            }
        )

        if should_try_recovery and orphan_index < len(orphan_first_names):
            first_name = orphan_first_names[orphan_index]

            if first_name not in words:
                new_name = _normalize_spaces(f"{first_name} {full_name}")

                record = _copy_record_with_updates(
                    record,
                    full_name=new_name,
                    confidence=0.85,
                    needs_review=True,
                    source=record.source,
                    extraction_method="orphan_first_name_recovered",
                )

                orphan_index += 1

        updated_records.append(record)

    return updated_records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _record_name_word_count(record: PersonRecord) -> int:
    if not record.full_name:
        return 0

    return len(record.full_name.split())


def _record_score(record: PersonRecord) -> int:
    score = 0

    if record.national_id:
        score += 10

    if record.registration_number:
        score += 10

    if record.source == "rules":
        score += 6

    if getattr(record, "extraction_method", None) == "structured_target_list":
        score += 5

    if getattr(record, "extraction_method", None) == "direct_legal_phrase":
        score += 4

    if record.needs_review is False:
        score += 2

    score += _record_name_word_count(record)

    return score


def _deduplicate_records(records: List[PersonRecord]) -> List[PersonRecord]:
    best_by_key = {}
    no_key_records: List[PersonRecord] = []

    for record in records:
        if record.national_id:
            key = f"id:{record.national_id}"
        elif record.registration_number:
            key = f"reg:{record.registration_number}"
        else:
            key = None

        if key:
            existing = best_by_key.get(key)

            if existing is None:
                best_by_key[key] = record
                continue

            if (
                existing.full_name
                and record.full_name
                and existing.full_name != record.full_name
            ):
                # Same ID, different names: likely two different people who
                # happen to share an (possibly OCR-misread) ID, not one
                # person detected twice. Keep both instead of silently
                # dropping one, and flag both since the ID collision itself
                # is worth a human double-checking.
                existing.needs_review = True
                record.needs_review = True
                no_key_records.append(record)
                continue

            if _record_score(record) > _record_score(existing):
                best_by_key[key] = record
        else:
            no_key_records.append(record)

    unique_records = list(best_by_key.values())
    seen_names = {record.full_name for record in unique_records if record.full_name}

    for record in no_key_records:
        if not record.full_name:
            continue

        if record.full_name in seen_names:
            continue

        if record.person_type != "Company":
            if not is_valid_person_name(record.full_name, has_national_id=False):
                continue

        seen_names.add(record.full_name)
        unique_records.append(record)

    return unique_records


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_person_candidates(cleaned_text: str) -> List[PersonRecord]:
    """Build PersonRecord candidates from rule-based extraction."""
    if cleaned_text is None:
        cleaned_text = ""

    records: List[PersonRecord] = []

    # Method 0: structured target list rows.
    structured_records = _extract_structured_list_records(cleaned_text)
    records.extend(structured_records)

    structured_ids = {
        record.national_id
        for record in structured_records
        if record.national_id
    }

    structured_registration_numbers = {
        record.registration_number
        for record in structured_records
        if record.registration_number
    }

    # Method 1: direct phrase + name + National ID.
    direct_records = _extract_direct_legal_phrase_records(cleaned_text)
    records.extend(direct_records)

    direct_ids = {
        record.national_id
        for record in direct_records
        if record.national_id
    } | structured_ids

    # Method 2: keyword lines.
    candidate_lines = extract_candidate_lines(cleaned_text)

    for line in candidate_lines:
        record = _build_record_from_line(line)

        if not record:
            continue

        if record.national_id and record.national_id in structured_ids:
            continue

        if record.registration_number and record.registration_number in structured_registration_numbers:
            continue

        records.append(record)

    # Method 3: fallback around National ID lines.
    context_records = _extract_context_records(
        cleaned_text,
        skip_ids=direct_ids,
    )
    records.extend(context_records)

    existing_ids = {
        record.national_id
        for record in records
        if record.national_id
    }

    # Method 4: emergency fallback.
    emergency_records = _extract_emergency_national_id_records(
        cleaned_text,
        skip_ids=existing_ids,
    )
    records.extend(emergency_records)

    final_records = _deduplicate_records(records)
    final_records = _recover_orphan_first_names(final_records, cleaned_text)

    return _deduplicate_records(final_records)