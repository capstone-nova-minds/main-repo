"""Rule-based extraction of person/company candidate records.

This service extracts persons and companies from normalized OCR text using
five methods, from most to least precise:

0. Structured target-list extraction (line-based logical records).
1. Direct legal-phrase + name + National ID regex.
2. Legal-keyword candidate lines.
3. Line-context extraction around every valid National ID.
4. Emergency wide-context fallback around National ID.
5. OCR orphan token recovery (first name for individuals, brand word for
   companies).

No LLM and no cloud AI are used.

Method 0 architecture (per-record, not whole-section):
    structured section detection
    -> extract section boundaries
    -> split section into logical records (line by line)
    -> parse every logical record independently
    -> detect identifiers from that same record
    -> extract name from that same record
    -> determine person type from that same record
    -> deduplicate

A logical record is never built by flattening the whole section into one
string and searching a character window around an identifier -- that is
what let a name from one row leak into a neighboring row's record. Instead
each record only ever sees its own lines.
"""

import re
from typing import Dict, List, Optional, Tuple

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
    COMPANY_NAME_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEADING_NOISE = re.compile(r"^[:\-\s،]+")
_MULTI_SPACE = re.compile(r"\s+")
_ARABIC_WORD_PATTERN = re.compile(r"[؀-ۿ]{2,}")

MIN_NAME_WORDS = 3


# Words that are never part of a person's/company's own name -- section
# labels, field labels, and generic document phrasing. Keeping this in one
# set means "شركة" (and friends) reliably breaks an *individual* name
# segment, and generic phrases like "يرجى اتخاذ الإجراءات اللازمة" can
# never be mistaken for a name.
_STRUCTURED_LABEL_NOISE = {
    "نوع",
    "الشخص",
    "فرد",
    "شركة",
    "مؤسسة",
    "جمعية",
    "بنك",
    "مدارس",
    "مستشفى",
    "مكتب",
    "مركز",
    "الرقم",
    "الوطني",
    "رقم",
    "وطني",
    "التسجيل",
    "تسجيل",
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
    "الموضوع",
    "الكتاب",
    "القضية",
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
        + r"(?P<name>[؀-ۿ\s]{5,120}?)\s*"
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


# ---------------------------------------------------------------------------
# National ID detection (supports OCR digit-group spacing + Arabic-Indic)
# ---------------------------------------------------------------------------

_ARABIC_INDIC_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def find_national_ids(text: str) -> List[str]:
    """Find every valid 11-digit National ID in text.

    Handles the common contiguous-digit case directly, and falls back to
    a digit-normalized pass (Arabic-Indic digits folded to ASCII, single
    spaces between digit groups collapsed) so OCR artifacts like
    "9 8 0 0 0..." or Arabic-Indic digits are still recognized -- without
    weakening the boundary check that rejects longer digit runs.
    """
    if not text:
        return []

    ids = list(NATIONAL_ID_PATTERN.findall(text))

    normalized = text.translate(_ARABIC_INDIC_TRANS)
    normalized = re.sub(r"(?<=[0-9])\s(?=[0-9])", "", normalized)

    for candidate in NATIONAL_ID_PATTERN.findall(normalized):
        if candidate not in ids:
            ids.append(candidate)

    return ids


def _first_national_id(line: str) -> Optional[str]:
    ids = find_national_ids(line)
    return ids[0] if ids else None


def is_valid_person_name(
    name: Optional[str],
    has_national_id: bool = False,
    direct_pattern_match: bool = False,
    min_words: int = MIN_NAME_WORDS,
) -> bool:
    """Reject false positives like headers/legal words.

    min_words lets a caller with a stronger trust signal than usual (e.g.
    Method 0's own verified-National-ID-anchored record) accept a shorter
    name than the general MIN_NAME_WORDS bar.
    """
    if not name:
        return False

    words = name.split()

    if any(word in _STRUCTURED_LABEL_NOISE for word in words):
        return False

    if any(word in INVALID_NAME_WORDS for word in words):
        return False

    if len(words) < min_words and not (has_national_id and direct_pattern_match):
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
    """Normalize company registration number.

    Supports:
    - REG-202608
    - REG - 202608
    - REG 202608
    - REG202608
    - 123456 (numeric-only registry number)
    """
    if not value:
        return None

    value = str(value).strip().upper()
    value = re.sub(r"\s*[-–—]\s*", "-", value)
    value = re.sub(r"\s+", "", value)

    letters_digits = re.fullmatch(r"([A-Z]{1,10})(\d{3,15})", value)

    if letters_digits:
        return f"{letters_digits.group(1)}-{letters_digits.group(2)}"

    if re.fullmatch(r"[A-Z]{1,10}-\d{3,15}", value):
        return value

    if re.fullmatch(r"\d{3,15}", value):
        return value

    return None


_REGISTRATION_VALUE_PATTERN = r"[A-Za-z]{1,10}\s*[-–—]?\s*\d{3,15}|\d{3,15}"

_REGISTRATION_LABEL_PATTERN = re.compile(
    r"رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة", flags=re.IGNORECASE
)
_REGISTRATION_VALUE_SEARCH_PATTERN = re.compile(
    r"[A-Za-z]{1,10}\s*[-–—]?\s*\d{3,15}"  # letter-prefixed, any digit length
    r"|(?<!\d)\d{3,10}(?!\d)"              # short numeric-only registry number
    r"|(?<!\d)\d{12,15}(?!\d)"             # long numeric-only, but not exactly
                                            # 11 digits -- that's a National ID
)


def _extract_registration_number(line: str) -> Optional[str]:
    """
    Extract company registration number.

    Supports:
    - رقم التسجيل: 123456
    - رقم التسجيل: REG-202608
    - REG-202608 : رقم التسجيل
    - REG-202608 : شركة المسار للتقنيات المالية رقم التسجيل
      (OCR put the company name *between* the value and its own label --
      the label search below doesn't require them to be adjacent)
    """
    if not line:
        return None

    patterns = [
        r"(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)\s*[:：]?\s*(?P<value>"
        + _REGISTRATION_VALUE_PATTERN
        + r")",
        r"(?P<value>"
        + _REGISTRATION_VALUE_PATTERN
        + r")\s*[:：]?\s*(?:رقم\s*التسجيل|سجل\s*تجاري|رقم\s*الشركة)",
    ]

    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)

        if match:
            return _normalize_registration_number(match.group("value"))

    # Fallback: label and value both present somewhere on the line, but not
    # immediately adjacent (OCR interposed the company name between them).
    if _REGISTRATION_LABEL_PATTERN.search(line):
        value_match = _REGISTRATION_VALUE_SEARCH_PATTERN.search(line)

        if value_match:
            return _normalize_registration_number(value_match.group(0))

    return None


# ---------------------------------------------------------------------------
# Method 0: Structured target list extraction (line-based logical records)
# ---------------------------------------------------------------------------

# Matches: "الأشخاص / الجهات المطلوبة", "الاشخاص الجهات المطلوبة",
# "الأشخاص المطلوبة", "الجهات المطلوبة", with or without hamza and with
# OCR spacing variations around the separator.
_SECTION_TITLE_PATTERN = re.compile(
    r"(?:الاشخاص|الأشخاص)\s*(?:[/|\\\-–—,،]\s*)?(?:الجهات\s*)?المطلوب[ةه]"
    r"|"
    r"الجهات\s*المطلوب[ةه]"
)


_SECTION_END_KEYWORDS = [
    "مرفق",
    "المرفقات",
    "وتفضلوا",
    "ختم",
    "كاتب",
    "صفحة",
    "يرجى اعتماد",
    "هذه الصفحة",
    "التوقيع",
    "رئيس المحكمة",
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
    Return only the lines belonging to the target persons/companies section.

    Structured extraction only runs when a real section title is found --
    it must never fall back to treating the whole document as the target
    section, since that is exactly what let unrelated text turn into
    bogus records.
    """
    if not cleaned_text:
        return []

    match = _SECTION_TITLE_PATTERN.search(cleaned_text)

    if not match:
        return []

    remainder = cleaned_text[match.end():]
    lines = [line.strip() for line in remainder.splitlines() if line.strip()]

    target_lines = []

    for line in lines:
        if _is_section_end(line):
            break

        if _is_recipient_or_greeting_line(line):
            continue

        target_lines.append(line)

    return target_lines


# Real Arabic personal names in this document set are consistently 3+
# characters (يوسف، سامر، فؤاد، ليث، عدي، سالم، ...) -- a bare 2-character
# fragment showing up as its own "word" is a truncated OCR artifact (a
# split glyph cluster like "قر"/"فر"), never a genuine name.
_MIN_INDIVIDUAL_WORD_LENGTH = 3


def _is_noise_word(word: str, allow_company_word: bool = False) -> bool:
    if allow_company_word and word in COMPANY_NAME_KEYWORDS:
        return False

    if not allow_company_word and len(word) < _MIN_INDIVIDUAL_WORD_LENGTH:
        return True

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
    text = re.sub(r"[A-Za-z]{1,10}\s*[-–—]?\s*\d{3,15}", " ", text)
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


def _clean_structured_person_name(raw_name: str, min_words: int = 3) -> Optional[str]:
    """Clean a person name captured from structured target-list rows.

    min_words defaults to 3 (the general "this reads like a real name"
    bar). Method 0 callers may lower it to 2 for a record that's already
    anchored by its own verified National ID -- the record boundary
    itself is the trust signal there, not the word count.
    """
    segments = _extract_name_segments(raw_name, allow_company_word=False)

    valid_segments = []

    for segment in segments:
        if len(segment) >= min_words:
            name = _normalize_spaces(" ".join(segment[-6:]))

            if is_valid_person_name(name, has_national_id=True, min_words=min_words):
                valid_segments.append(name)

    if not valid_segments:
        return None

    return valid_segments[-1]


def _extract_name_near_national_id(
    section_text: str,
    national_id: str,
    min_words: int = 3,
) -> Optional[str]:
    """
    Extract individual name around National ID from flattened target section.

    Handles OCR order issues:
    - name before ID
    - name after ID
    - fields split into different lines
    """
    if not section_text or not national_id:
        return None

    id_pos = section_text.find(national_id)

    if id_pos == -1:
        return None

    before_id = section_text[max(0, id_pos - 220):id_pos]
    after_id = section_text[id_pos + len(national_id): id_pos + len(national_id) + 180]

    # Usually Arabic form has name before "الرقم الوطني".
    before_name = _clean_structured_person_name(before_id, min_words=min_words)

    if before_name:
        return before_name

    after_name = _clean_structured_person_name(after_id, min_words=min_words)

    if after_name:
        return after_name

    return None


# ---------------------------------------------------------------------------
# Method 0: logical-record splitting
# ---------------------------------------------------------------------------

# Explicit "نوع الشخص: فرد/شركة" declaration -- only counts as an anchor
# when it actually carries a value, an empty/truncated label is not one.
_TYPE_DECLARATION_PATTERN = re.compile(r"نوع\s*الشخص\s*[:：\-]?\s*(فرد|شركة)")


def _line_explicit_type(line: str) -> Optional[str]:
    match = _TYPE_DECLARATION_PATTERN.search(line)

    if not match:
        return None

    return "Company" if match.group(1) == "شركة" else "Individual"


class _LogicalRecord:
    """One row of the structured target list, built only from its own lines."""

    __slots__ = ("national_id", "registration_number", "explicit_type", "raw_lines")

    def __init__(self) -> None:
        self.national_id: Optional[str] = None
        self.registration_number: Optional[str] = None
        self.explicit_type: Optional[str] = None
        self.raw_lines: List[str] = []

    def has_identifier(self) -> bool:
        return bool(self.national_id or self.registration_number)

    @property
    def text(self) -> str:
        return " ".join(self.raw_lines)

    def _is_anchor_line(self, line: str) -> bool:
        return bool(
            _first_national_id(line)
            or _extract_registration_number(line)
            or _line_explicit_type(line)
        )

    @property
    def individual_text(self) -> str:
        """Same lines as `text`, but with plain (non-anchor) name-fragment
        lines reordered so a lone one-word line never sits ahead of a
        multi-word line.

        EasyOCR sometimes emits a short single-word row (typically a
        family-name fragment on its own bounding box) *before* the row it
        visually belongs after, e.g. "الخطيب" then "ليث محمود" instead of
        "ليث محمود" then "الخطيب". Moving one-word fragment lines after
        multi-word ones restores natural given-name -> family-name order
        without ever reordering the words *within* a single OCR line.

        When a record has *no* multi-word line at all -- every name word
        landed on its own separate single-word line -- there's no anchor
        to move fragments after, but the same left-to-right/right-to-left
        sorting artifact still applies directly: two consecutive
        single-word boxes on one physical row get listed in reverse
        reading order (e.g. "حمد" then "وائل" instead of "وائل" then
        "حمد"). Reversing the whole single-word group in that case
        restores correct order; it's a no-op for the too-short (<3 char)
        fragments among them since those get filtered out as noise
        wherever they land.
        """
        anchor_lines = []
        plain_lines = []

        for line in self.raw_lines:
            if self._is_anchor_line(line):
                anchor_lines.append(line)
            else:
                plain_lines.append(line)

        multi_word_lines = [
            line for line in plain_lines
            if len(_ARABIC_WORD_PATTERN.findall(line)) >= 2
        ]
        single_word_lines = [
            line for line in plain_lines
            if len(_ARABIC_WORD_PATTERN.findall(line)) == 1
        ]
        other_lines = [
            line for line in plain_lines
            if line not in multi_word_lines and line not in single_word_lines
        ]

        if not multi_word_lines and len(single_word_lines) >= 2:
            single_word_lines = list(reversed(single_word_lines))

        return " ".join(anchor_lines + multi_word_lines + single_word_lines + other_lines)

    @property
    def has_isolated_family_fragment(self) -> bool:
        """
        True when this record's own lines already confirm which captured
        word is the family name -- i.e. OCR split the row into a lone
        single-word line *and* a separate multi-word line (the case
        `individual_text` reorders). That confirms the single word is the
        trailing family name and any still-missing word is an *interior*
        gap (the grandfather slot, right before it).

        False when the record's own lines merged straight into one
        contiguous multi-word capture with no isolated single-word
        fragment -- here a missing word is a *leading* gap (the record
        was cut off at its start, most commonly dropping the given name),
        so it belongs at the front instead.
        """
        anchor_lines = []
        plain_lines = []

        for line in self.raw_lines:
            if self._is_anchor_line(line):
                anchor_lines.append(line)
            else:
                plain_lines.append(line)

        has_single = any(len(_ARABIC_WORD_PATTERN.findall(line)) == 1 for line in plain_lines)
        has_multi = any(len(_ARABIC_WORD_PATTERN.findall(line)) >= 2 for line in plain_lines)

        return has_single and has_multi

    @property
    def has_multi_word_line(self) -> bool:
        """
        True when at least one of this record's own plain lines held 2+
        words together (e.g. "فؤاد العزام"). A multi-word line is a
        single OCR detection, so it reliably preserves true reading order
        -- a captured 2-word pair built *from one* is trustworthy enough
        to safely assume it's the trailing (grandfather+family) pair.

        False when every plain line held exactly one word each: two
        separate single-word boxes carry no guarantee about which slots
        they fill (leading, trailing, or interior) -- see
        _recover_orphan_first_names' "unconfirmed pair" handling.
        """
        return any(
            len(_ARABIC_WORD_PATTERN.findall(line)) >= 2
            for line in self.raw_lines
            if not self._is_anchor_line(line)
        )

    @property
    def short_fragment_candidates(self) -> List[str]:
        """
        Standalone 2-character plain-line fragments -- too short to trust
        as a name on their own (see _MIN_INDIVIDUAL_WORD_LENGTH), but
        exactly the shape of one half of a word EasyOCR split across two
        bounding boxes (e.g. "قراس" -> "قر" left in this record, "راس"
        landing separately in the footer as an orphan token). Kept, in
        order, purely as raw material for _stitch_recovered_token to try
        recombining with an orphan token -- never used as a name on their
        own.
        """
        fragments = []

        for line in self.raw_lines:
            if self._is_anchor_line(line):
                continue

            words = _ARABIC_WORD_PATTERN.findall(line)

            if len(words) == 1 and len(words[0]) == 2:
                fragments.append(words[0])

        return fragments


def _split_into_logical_records(target_lines: List[str]) -> List[_LogicalRecord]:
    """
    Group target-section lines into logical records.

    A new record starts at an identifier line (national ID or registration
    number) once the current record already has an identifier of its own --
    that is the "close the previous record" rule. A bare explicit-type
    declaration ("نوع الشخص: شركة") with no identifier yet does not close
    anything by itself; it just marks the type of whatever record is being
    built, so a registration number arriving on the very next physical line
    (an OCR row split into two lines) still lands in the same record.
    """
    records: List[_LogicalRecord] = []
    current: Optional[_LogicalRecord] = None

    for line in target_lines:
        national_id = _first_national_id(line)
        registration_number = _extract_registration_number(line)
        explicit_type = _line_explicit_type(line)
        is_anchor = bool(national_id or registration_number or explicit_type)

        if is_anchor:
            if current is not None and current.has_identifier():
                records.append(current)
                current = _LogicalRecord()
            elif current is None:
                current = _LogicalRecord()

            if national_id and not current.national_id:
                current.national_id = national_id

            if registration_number and not current.registration_number:
                current.registration_number = registration_number

            if explicit_type and not current.explicit_type:
                current.explicit_type = explicit_type

            current.raw_lines.append(line)
        else:
            if current is None:
                current = _LogicalRecord()

            current.raw_lines.append(line)

    if current is not None:
        records.append(current)

    return records


# ---------------------------------------------------------------------------
# Method 0: company name reconstruction
# ---------------------------------------------------------------------------

def _build_company_name_from_record(record: _LogicalRecord) -> Optional[str]:
    """
    Build a company name from a logical record's own lines only.

    Handles OCR damage seen in practice:
    - the company keyword ("شركة") ends up trailing the rest of the name
      instead of leading it (OCR line-order artifact) -> reordered to the
      front.
    - the company keyword never got OCR'd into the name text at all
      (it only appeared in the "نوع الشخص: شركة" declaration) -> the
      keyword is still prefixed so the name never starts mid-phrase.
    """
    segments = _extract_name_segments(record.text, allow_company_word=True)
    segments = [
        segment for segment in segments
        if len(segment) >= 2 or segment[0] in COMPANY_NAME_KEYWORDS
    ]
    words: List[str] = [word for segment in segments for word in segment]
    words = _dedupe_repeated_company_keyword(words)

    if not words:
        return None

    name = _ensure_company_keyword_prefix(_normalize_spaces(" ".join(words)))

    return name or None


def _dedupe_repeated_company_keyword(words: List[str]) -> List[str]:
    """OCR sometimes detects the company keyword twice for one row (once
    from a "نوع الشخص: شركة" declaration, once as part of the name itself).
    Keep only the first occurrence -- a company name should carry its
    keyword exactly once."""
    seen_keyword = False
    deduped = []

    for word in words:
        if word in COMPANY_NAME_KEYWORDS:
            if seen_keyword:
                continue
            seen_keyword = True

        deduped.append(word)

    return deduped


def _ensure_company_keyword_prefix(name: str) -> str:
    """Make sure a company name starts with its own keyword, in order.

    Never reverses the rest of the name -- only relocates (or, if missing
    entirely, prepends) the single company keyword word.
    """
    words = name.split()

    if not words:
        return name

    for keyword in COMPANY_NAME_KEYWORDS:
        if keyword in words:
            if words[0] != keyword:
                words.remove(keyword)
                words.insert(0, keyword)
            return _normalize_spaces(" ".join(words))

    # No company keyword survived in the name text at all -- the record is
    # only known to be a company because of its registration number / type
    # declaration, so prefix the generic keyword rather than emit a name
    # that doesn't read as a company.
    return _normalize_spaces("شركة " + name)


def _company_missing_brand_word(name: Optional[str]) -> bool:
    """
    True when a company name looks like "شركة للخدمات المالية" -- the
    keyword and the service-type phrase are both present and in order,
    but the brand word that normally sits between them was dropped by OCR.
    """
    if not name:
        return False

    words = name.split()

    if len(words) < 2 or words[0] not in COMPANY_NAME_KEYWORDS:
        return False

    return bool(re.match(r"^ل[؀-ۿ]", words[1]))


# ---------------------------------------------------------------------------
# Method 0: person type + confidence
# ---------------------------------------------------------------------------

def _determine_person_type(record: _LogicalRecord, name: Optional[str]) -> str:
    """
    Priority (evaluated only against this same logical record):
    1. registration_number exists -> Company
    2. explicit "نوع الشخص: شركة" -> Company
    3. national_id exists -> Individual
    4. explicit "نوع الشخص: فرد" -> Individual
    5. company keyword in the name -> Company
    6. otherwise -> Individual (caller marks needs_review)
    """
    if record.registration_number:
        return "Company"

    if record.explicit_type == "Company":
        return "Company"

    if record.national_id:
        return "Individual"

    if record.explicit_type == "Individual":
        return "Individual"

    if name and is_company_line(name):
        return "Company"

    return "Individual"


def _build_structured_record_from_logical(
    record: _LogicalRecord,
) -> Tuple[Optional[PersonRecord], List[str]]:
    """Returns (record_or_none, short_fragment_candidates) -- the latter is
    only ever non-empty for an Individual record, and is raw material for
    later orphan-token stitching, never used to build full_name here."""
    person_type = _determine_person_type(record, None)

    if person_type == "Company":
        full_name = _build_company_name_from_record(record)

        if not full_name or not record.registration_number:
            # A company without a registration number and without a
            # reconstructable name is not a usable structured record --
            # let the other extraction methods have a chance instead of
            # emitting a guess.
            if not full_name:
                return None, []

        return _make_person_record(
            full_name=full_name,
            national_id=None,
            registration_number=record.registration_number,
            person_type="Company",
            confidence=EXACT_MATCH_CONFIDENCE + 0.05,
            needs_review=False,
            source="rules",
            extraction_method="structured_target_list",
        ), []

    # Individual.
    individual_text = record.individual_text
    capture_tier = "strict"

    if record.national_id:
        full_name = _extract_name_near_national_id(individual_text, record.national_id)

        if not full_name:
            # Fall back to a 2-word capture -- the record is still trusted
            # because it's anchored by its own verified National ID; a
            # missing given/father name is recovered later from the
            # orphan-token pool, not guessed here.
            full_name = _extract_name_near_national_id(
                individual_text, record.national_id, min_words=2
            )
            capture_tier = "relaxed"

        if not full_name:
            # Last resort: even a single genuine word anchored by a
            # verified National ID is worth surfacing rather than silently
            # dropping the record. Recovery still gets attempted for this
            # tier (see _recover_orphan_first_names), but only ever lands
            # on a confident result when the orphan pool cleanly supplies
            # the rest -- otherwise it stays flagged for review.
            full_name = _extract_name_near_national_id(
                individual_text, record.national_id, min_words=1
            )
            capture_tier = "sparse"
    else:
        full_name = _clean_structured_person_name(individual_text)

    if not full_name:
        return None, []

    if not record.national_id and record.explicit_type != "Individual":
        # No identifier and no explicit type declaration -- not a valid
        # structured record on its own.
        return None, []

    if capture_tier == "sparse":
        confidence, needs_review = 0.60, True
    elif capture_tier == "relaxed":
        confidence, needs_review = 0.75, True
    else:
        confidence, needs_review = EXACT_MATCH_CONFIDENCE + 0.05, False

    # A still-incomplete name (< 4 words) may get one more word filled in
    # later by orphan-token recovery. Whether that recovered word belongs
    # at the front or in the interior (just before the family name)
    # depends on how *this* record's own lines were shaped -- tag it now
    # while we still have that structural information, since orphan
    # recovery only ever sees the flat PersonRecord afterwards.
    extraction_method = "structured_target_list"

    if len(full_name.split()) < _FULL_NAME_WORD_COUNT and record.has_isolated_family_fragment:
        extraction_method = "structured_target_list_reordered"
    elif capture_tier == "relaxed" and not record.has_multi_word_line:
        # A 2-word capture built from two *separate* single-word lines
        # (not one multi-word line) carries no guarantee it's the
        # trailing pair -- it could just as easily be an interior pair
        # with gaps on both sides. Confidently prepending both recovered
        # words assumes the former; when it's actually the latter, that
        # produces a confident wrong name instead of a merely incomplete
        # one. Tag it so recovery treats this pair as unconfirmed and
        # leaves it as its honest partial capture rather than guessing.
        extraction_method = "structured_target_list_unconfirmed_pair"

    person_record = _make_person_record(
        full_name=full_name,
        national_id=record.national_id,
        registration_number=None,
        person_type="Individual",
        confidence=confidence,
        needs_review=needs_review,
        source="rules",
        extraction_method=extraction_method,
    )

    return person_record, record.short_fragment_candidates


def _extract_structured_list_records(
    cleaned_text: str,
) -> Tuple[List[PersonRecord], Dict[str, List[str]]]:
    """
    Extract target rows from:
    الأشخاص / الجهات المطلوبة:
    - name | الرقم الوطني: id | نوع الشخص: فرد
    - company name | رقم التسجيل: REG-202608 | نوع الشخص: شركة

    Each row is parsed as its own logical record (its own lines only) so a
    name, identifier, or type keyword from one row can never bleed into
    another row's record.

    Returns (records, stitch_hints) -- stitch_hints maps a still-incomplete
    Individual record's own National ID to its record's short (2-char)
    leftover fragments, for _recover_orphan_first_names to try recombining
    with an orphan token later.
    """
    if not cleaned_text:
        return [], {}

    target_lines = _get_target_section_lines(cleaned_text)

    if not target_lines:
        return [], {}

    logical_records = _split_into_logical_records(target_lines)

    records: List[PersonRecord] = []
    stitch_hints: Dict[str, List[str]] = {}

    for logical_record in logical_records:
        record, short_fragments = _build_structured_record_from_logical(logical_record)

        if record:
            records.append(record)

            if record.national_id and short_fragments:
                stitch_hints[record.national_id] = short_fragments

    return _deduplicate_records(records), stitch_hints


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
    seen_ids = set()

    for pattern in _DIRECT_PHRASE_PATTERNS:
        for match in pattern.finditer(search_text):
            national_id = match.group("id")

            if national_id in seen_ids:
                continue

            name = _clean_direct_match_name(match.group("name"))

            if not name:
                continue

            seen_ids.add(national_id)

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
# OCR orphan token recovery
# ---------------------------------------------------------------------------
# EasyOCR sometimes detects one visual row's leading word as a separate
# bounding box that ends up sorted far away from its row -- typically down
# in the footer/signature area. This recovers exactly that single word and
# reattaches it to the record it visually belongs to, never invents a word
# that isn't present verbatim in the OCR text, and never touches a record
# that already looks complete.

# Generic words/labels that can show up as a lone stray OCR line but are
# never themselves a name fragment -- section labels, footer/signature
# boilerplate, and document furniture. Eligibility for the orphan pool is
# a deny-list (not an allow-list of "known first names"): Jordanian given
# names can't be enumerated in advance, but the generic document
# vocabulary that leaks onto its own line can.
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
    "لديكم",
    "افي",
    "في",
    "الأردنية",
    "الاردنية",
    "عمان",
    "التوقيع",
    "كتاب",
    "الأصول",
    "الاصول",
    "حول",
    "بقبول",
    "الاحترام",
    "الاحتر",
    "قرار",
    "يرجى",
    "اتخاذ",
    "حسب",
}


def _is_eligible_orphan_token(word: str) -> bool:
    if len(word) < _MIN_INDIVIDUAL_WORD_LENGTH:
        return False

    return not (
        word in _STRUCTURED_LABEL_NOISE
        or word in INVALID_NAME_WORDS
        or word in _NAME_NOISE_WORDS
        or word in _ORPHAN_NOISE_WORDS
        or word in COMPANY_NAME_KEYWORDS
    )


def _get_lines_after_section_end(cleaned_text: str) -> List[str]:
    """
    Lines strictly *after* the structured section's footer boundary --
    exactly the part `_get_target_section_lines` excludes. This is where
    EasyOCR's stray single-word fragments (scattered given names, brand
    words) actually land in both documents this system has seen so far.

    Restricting the scan to here (instead of rescanning from the section
    title onward) means a word that's already inside some record's own
    lines can never be double-counted as an "orphan" too.
    """
    if not cleaned_text:
        return []

    match = _SECTION_TITLE_PATTERN.search(cleaned_text)

    if not match:
        return []

    remainder = cleaned_text[match.end():]
    lines = [line.strip() for line in remainder.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if _is_section_end(line):
            return lines[i:]

    return []


def _extract_orphan_single_words(cleaned_text: str) -> List[str]:
    """
    Standalone single-Arabic-word lines from past the section's footer
    boundary, in document order. These are the candidate pool for orphan
    recovery.
    """
    candidates: List[str] = []

    for line in _get_lines_after_section_end(cleaned_text):
        words = _ARABIC_WORD_PATTERN.findall(line)

        if len(words) != 1:
            continue

        word = _fix_ocr_name_word(words[0])

        if not _is_eligible_orphan_token(word):
            continue

        candidates.append(word)

    return candidates


_RECOVERABLE_INDIVIDUAL_METHODS = {
    "structured_target_list",
    "structured_target_list_reordered",
    "structured_target_list_unconfirmed_pair",
    "national_id_context",
    "emergency_national_id_context",
}

# Arrangement is never confidently applied for this tag (see
# _build_structured_record_from_logical) -- it still consumes its share
# of the orphan pool to keep alignment correct for later records, but
# always leaves the record as its own honest partial capture.
_UNCONFIRMED_PAIR_METHOD = "structured_target_list_unconfirmed_pair"

# Target shape for a full Arabic legal name: [given, father, grandfather,
# family]. The family name is always last (individual records are already
# reordered so a lone family-name fragment sits at the end -- see
# _LogicalRecord.individual_text), so however many words are still
# missing are the *leading* slots.
_FULL_NAME_WORD_COUNT = 4


def _pop_orphan_tokens(orphan_pool: List[str], count: int) -> Optional[List[str]]:
    """Pop the next `count` eligible tokens from the pool, in order.
    Returns None (and leaves the pool untouched) if there aren't enough."""
    indices = []

    for i, token in enumerate(orphan_pool):
        if len(indices) == count:
            break

        indices.append(i)

    if len(indices) < count:
        return None

    tokens = [orphan_pool[i] for i in indices]

    for i in reversed(indices):
        orphan_pool.pop(i)

    return tokens


# Above this many missing words, a *plain* prepend/interior-insert stops
# being a safe default -- see _recover_orphan_first_names. Missing==3 is
# still attempted, but only with the more specific "single interior word"
# arrangement below, and always landing at a slightly lower confidence.
_MAX_SAFE_RECOVERY_GAP = 2
_MAX_RECOVERY_GAP = 3


def _stitch_recovered_tokens(tokens: List[str], fragments: List[str]) -> List[str]:
    """
    Try to extend each recovered orphan token using one of this record's
    own leftover 2-character fragments (see
    _LogicalRecord.short_fragment_candidates) -- e.g. record fragment
    "قر" + orphan token "راس" -> "قراس". Only merges when the fragment's
    last character equals the token's first character, which is exactly
    the shape produced when EasyOCR splits one cursive-joined word across
    two bounding boxes and both boxes pick up the shared connecting
    letter. Never invents a character neither piece already has, and
    each fragment is used for at most one token.
    """
    if not fragments:
        return tokens

    available = list(fragments)
    stitched = []

    for token in tokens:
        match_index = next(
            (i for i, fragment in enumerate(available) if fragment[-1] == token[0]),
            None,
        )

        if match_index is not None:
            fragment = available.pop(match_index)
            stitched.append(fragment + token[1:])
        else:
            stitched.append(token)

    return stitched


def _recover_orphan_first_names(
    records: List[PersonRecord],
    orphan_pool: List[str],
    stitch_hints: Dict[str, List[str]],
) -> List[PersonRecord]:
    """Fill in a missing name slot for an otherwise-anchored Individual
    record -- never touches Company records, never runs on an
    already-complete 4-word name.

    - Missing exactly 1 word: could be the *interior* grandfather slot
      (when the record's own lines already confirmed a lone family-name
      fragment, tagged "structured_target_list_reordered" -- the recovered
      word slots in right before the family name) or a *leading* slot cut
      off the front of a single contiguous capture (the far more common
      case -- the recovered word is prepended).
    - Missing 2 words: the record only kept the trailing pair (e.g.
      grandfather+family); EasyOCR emits same-row RTL word fragments in
      left-to-right (i.e. reversed) order, so the recovered group is
      reversed back to correct reading order before being prepended.
    - Missing 3 words (only one word of its own survived, and that word's
      position within the name is otherwise unknown): reversing the 3
      recovered tokens gives [given, father, family] in correct reading
      order; the one surviving word is inserted between father and
      family (the grandfather slot) -- the same interior position a
      lone surviving word would occupy in the missing==1 case. This is a
      best-effort arrangement (lower confidence than the missing<=2
      cases), since -- unlike missing<=2 -- there's no direct structural
      confirmation that the surviving word truly sits there.
    - Beyond that, the record is left as its own honest (already
      needs_review) partial capture -- but the tokens it *would* have
      needed are still popped from the pool and discarded rather than
      left in place. The pool is a flat, whole-document scan, not grouped
      per record; if a record doesn't consume its share, the next record
      in line would wrongly inherit tokens that were actually this one's,
      shifting every downstream recovery by one.
    """
    updated_records: List[PersonRecord] = []

    for record in records:
        full_name = record.full_name or ""
        words = full_name.split()
        missing = _FULL_NAME_WORD_COUNT - len(words)
        extraction_method = getattr(record, "extraction_method", None)

        should_try_recovery = (
            record.person_type == "Individual"
            and bool(record.national_id)
            and missing >= 1
            and extraction_method in _RECOVERABLE_INDIVIDUAL_METHODS
        )

        if should_try_recovery:
            recovered = _pop_orphan_tokens(orphan_pool, missing)

            if (
                recovered
                and missing <= _MAX_RECOVERY_GAP
                and extraction_method != _UNCONFIRMED_PAIR_METHOD
                and not any(token in words for token in recovered)
            ):
                fragments = stitch_hints.get(record.national_id, [])
                confidence = 0.90

                if missing == 1 and extraction_method == "structured_target_list_reordered":
                    recovered = _stitch_recovered_tokens(recovered, fragments)
                    new_words = words[:-1] + recovered + words[-1:]
                elif missing == 1:
                    recovered = _stitch_recovered_tokens(recovered, fragments)
                    new_words = recovered + words
                elif missing == 2:
                    ordered = _stitch_recovered_tokens(list(reversed(recovered)), fragments)
                    new_words = ordered + words
                else:
                    # missing == 3: [given, father, <surviving word>, family]
                    ordered = _stitch_recovered_tokens(list(reversed(recovered)), fragments)
                    new_words = [ordered[0], ordered[1]] + words + [ordered[2]]
                    confidence = 0.80

                record = _copy_record_with_updates(
                    record,
                    full_name=_normalize_spaces(" ".join(new_words)),
                    confidence=confidence,
                    needs_review=False,
                    source=record.source,
                    extraction_method="orphan_first_name_recovered",
                )
            # else: tokens were consumed above (or unavailable) to keep the
            # pool aligned for later records, but not trusted enough to
            # apply -- the record keeps its own honest partial capture.

        updated_records.append(record)

    return updated_records


def _recover_orphan_company_brand_words(
    records: List[PersonRecord],
    orphan_pool: List[str],
) -> List[PersonRecord]:
    """Insert a missing brand word into an otherwise-complete Company
    record (e.g. "شركة للخدمات المالية" -> "شركة الأفق للخدمات المالية").

    Never runs on Individual records, never invents a word not already
    present verbatim as a standalone OCR line, and is skipped entirely if
    the company name does not show the "keyword + missing brand" shape.
    """
    updated_records: List[PersonRecord] = []

    for record in records:
        if (
            record.person_type == "Company"
            and getattr(record, "extraction_method", None) == "structured_target_list"
            and _company_missing_brand_word(record.full_name)
        ):
            recovered = _pop_orphan_tokens(orphan_pool, 1)

            if recovered:
                brand_word = recovered[0]
                words = record.full_name.split()
                new_name = _normalize_spaces(" ".join([words[0], brand_word] + words[1:]))

                record = _copy_record_with_updates(
                    record,
                    full_name=new_name,
                    confidence=0.90,
                    needs_review=False,
                    extraction_method="orphan_company_brand_recovered",
                )

        updated_records.append(record)

    return updated_records


def _recover_orphan_tokens(
    records: List[PersonRecord],
    cleaned_text: str,
    stitch_hints: Dict[str, List[str]],
) -> List[PersonRecord]:
    orphan_pool = _extract_orphan_single_words(cleaned_text)

    if not orphan_pool:
        return records

    records = _recover_orphan_first_names(records, orphan_pool, stitch_hints)
    records = _recover_orphan_company_brand_words(records, orphan_pool)

    return records


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
            if key not in best_by_key:
                best_by_key[key] = record
                continue

            if _record_score(record) > _record_score(best_by_key[key]):
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
    structured_records, stitch_hints = _extract_structured_list_records(cleaned_text)
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
    final_records = _recover_orphan_tokens(final_records, cleaned_text, stitch_hints)

    return _deduplicate_records(final_records)
