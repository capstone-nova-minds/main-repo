"""Rule-based extraction of person/company candidate records.

This service extracts persons and companies from normalized OCR text using
four methods, from most to least precise:

1. Direct legal-phrase + name + National ID regex.
2. Legal-keyword candidate lines.
3. Line-context extraction around every valid National ID.
4. Emergency wide-context fallback around National ID.

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
}


_NAME_NOISE_WORDS = {
    "رقم",
    "وطني",
    "دعوى",
    "بداية",
    "حقوق",
    "اربد",
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
        + r"(?P<name>[\u0600-\u06FF\s]{5,120}?)\s*"
        + _ID_CONNECTOR
        + r"\s*[:\-]?\s*[\(\[]?\s*"
        + r"(?P<id>\d{11})"
        + r"\s*[\)\]]?"
    )
    for phrase in _DIRECT_PHRASE_KEYWORDS
]


# Common OCR mistakes in Arabic names.
_OCR_NAME_FIXES = {
    "بوسف": "يوسف",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_kwargs(**kwargs):
    """Only pass fields that exist in PersonRecord."""
    try:
        fields = set(PersonRecord.model_fields.keys())  # Pydantic v2
    except AttributeError:
        fields = set(PersonRecord.__fields__.keys())  # Pydantic v1

    return {key: value for key, value in kwargs.items() if key in fields}


def _make_person_record(**kwargs) -> PersonRecord:
    return PersonRecord(**_record_kwargs(**kwargs))


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
    """Reject false positives like land/property descriptions or legal terms."""
    if not name:
        return False

    words = name.split()

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


def _extract_registration_number(line: str) -> Optional[str]:
    patterns = [
        r"رقم تسجيل\s*[:：]?\s*(\d{3,15})",
        r"سجل تجاري\s*[:：]?\s*(\d{3,15})",
        r"رقم الشركة\s*[:：]?\s*(\d{3,15})",
    ]

    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Method 1: Direct legal phrase + name + National ID
# ---------------------------------------------------------------------------

def _clean_direct_match_name(raw_name: str) -> Optional[str]:
    text = _cut_before_stop_words(raw_name)
    words = _ARABIC_WORD_PATTERN.findall(text)
    words = _clean_words(words)

    if not words:
        return None

    name = " ".join(words).strip()
    name = _normalize_spaces(name)

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

    name = " ".join(words).strip()
    name = _normalize_spaces(name)

    if len(name) < 5:
        return None

    return name


def _build_record_from_line(line: str) -> Optional[PersonRecord]:
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
    """Extract name from previous + current + next line around a National ID."""
    line = lines[line_index]
    ids_on_line = find_national_ids(line)

    if not ids_on_line:
        return None

    national_id = ids_on_line[0]

    prev_line = lines[line_index - 1] if line_index > 0 else ""
    next_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""

    context = f"{prev_line} {line} {next_line}"
    context = context.replace("(", " ").replace(")", " ")
    context = context.replace("،", " ")
    context = _normalize_spaces(context)

    id_pos = context.find(national_id)

    if id_pos == -1:
        return None

    before_id = context[:id_pos].strip()

    name_part = before_id

    for hint in _PERSON_HINTS:
        if hint in before_id:
            name_part = before_id.split(hint, 1)[1]
            break
    else:
        fallback_words = _ARABIC_WORD_PATTERN.findall(before_id)
        name_part = " ".join(fallback_words[-8:])

    name_part = _cut_before_stop_words(name_part)

    words = _ARABIC_WORD_PATTERN.findall(name_part)
    words = _clean_words(words)

    if len(words) < MIN_NAME_WORDS:
        return None

    words = words[-5:]

    name = " ".join(words).strip()
    name = _normalize_spaces(name)

    if not is_valid_person_name(name, has_national_id=True):
        return None

    return name, national_id


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
    """
    Relaxed validation for emergency fallback.

    Used only when a valid National ID exists but stricter extraction methods
    failed. Allows 2+ Arabic words because OCR may miss part of the full name.
    """
    if not name:
        return False

    words = name.split()

    if len(words) < 2:
        return False

    if any(word in INVALID_NAME_WORDS for word in words):
        return False

    if all(word in _NAME_NOISE_WORDS for word in words):
        return False

    return True


def _extract_emergency_national_id_records(
    cleaned_text: str,
    skip_ids: set,
) -> List[PersonRecord]:
    """
    Last-resort extraction.

    Used when:
    - National ID exists in OCR text
    - direct phrase extraction failed
    - keyword-line extraction failed
    - line-context extraction failed

    Strategy:
    Take a wider text window before each National ID and recover the closest
    Arabic name before the ID. Result is marked needs_review=True.
    """
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

        hint_positions = []

        for hint in _PERSON_HINTS:
            pos = before_id.rfind(hint)
            if pos != -1:
                hint_positions.append((pos, hint))

        found_person_hint = False

        if hint_positions:
            pos, hint = sorted(hint_positions, key=lambda x: x[0])[-1]
            name_part = before_id[pos + len(hint):]
            found_person_hint = True
        else:
            name_part = before_id

        name_part = _cut_before_stop_words(name_part)

        name_part = name_part.replace("(", " ")
        name_part = name_part.replace(")", " ")
        name_part = name_part.replace("[", " ")
        name_part = name_part.replace("]", " ")
        name_part = name_part.replace("،", " ")
        name_part = name_part.replace(":", " ")
        name_part = name_part.replace("-", " ")
        name_part = _normalize_spaces(name_part)

        words = _ARABIC_WORD_PATTERN.findall(name_part)
        words = _clean_words(words)

        if not words:
            continue

        segments = []
        current_segment = []

        for word in words:
            if word in INVALID_NAME_WORDS:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue

            current_segment.append(word)

        if current_segment:
            segments.append(current_segment)

        valid_segments = []

        for segment in segments:
            min_words = 2 if found_person_hint else 3

            if len(segment) >= min_words:
                valid_segments.append(segment)

        if not valid_segments:
            continue

        selected_words = valid_segments[-1]
        selected_words = selected_words[-5:]

        full_name = " ".join(selected_words).strip()
        full_name = _normalize_spaces(full_name)

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

    if record.source == "rules":
        score += 6

    if getattr(record, "extraction_method", None) == "direct_legal_phrase":
        score += 4

    if record.needs_review is False:
        score += 2

    score += _record_name_word_count(record)

    return score


def _deduplicate_records(records: List[PersonRecord]) -> List[PersonRecord]:
    best_by_id = {}
    no_id_records: List[PersonRecord] = []

    for record in records:
        if record.national_id:
            key = record.national_id

            if key not in best_by_id:
                best_by_id[key] = record
                continue

            if _record_score(record) > _record_score(best_by_id[key]):
                best_by_id[key] = record
        else:
            no_id_records.append(record)

    unique_records = list(best_by_id.values())
    seen_names = {record.full_name for record in unique_records if record.full_name}

    for record in no_id_records:
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

    # Method 1: direct phrase + name + National ID.
    direct_records = _extract_direct_legal_phrase_records(cleaned_text)
    records.extend(direct_records)

    direct_ids = {
        record.national_id
        for record in direct_records
        if record.national_id
    }

    # Method 2: keyword lines.
    candidate_lines = extract_candidate_lines(cleaned_text)

    for line in candidate_lines:
        record = _build_record_from_line(line)

        if record:
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

    return _deduplicate_records(records)