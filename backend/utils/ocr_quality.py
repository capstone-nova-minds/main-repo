"""OCR quality scoring, used by the OCR router to pick the best engine result."""

import re

ARABIC_CHAR_PATTERN = re.compile(r"[؀-ۿ]")

EXPECTED_LEGAL_KEYWORDS = [
    "محكمة",
    "قضية",
    "تنفيذ",
    "حجز",
    "المدين",
    "المحكوم عليه",
    "رقم",
    "تاريخ",
]

MIN_USEFUL_TEXT_LENGTH = 20


def calculate_ocr_quality(text: str, average_confidence: float) -> float:
    """Score OCR output from 0.0 (unusable) to 1.0 (high quality).

    Combines Arabic character density, presence of expected legal
    keywords, line count, and the engine's own reported confidence.
    """
    if not text or not text.strip():
        return 0.0

    text = text.strip()
    total_chars = len(text)
    arabic_chars = len(ARABIC_CHAR_PATTERN.findall(text))
    arabic_ratio = arabic_chars / total_chars if total_chars else 0.0

    keyword_matches = sum(1 for kw in EXPECTED_LEGAL_KEYWORDS if kw in text)
    keyword_score = min(keyword_matches / len(EXPECTED_LEGAL_KEYWORDS), 1.0)

    line_count = len([line for line in text.splitlines() if line.strip()])
    line_score = min(line_count / 10, 1.0)

    confidence_score = max(0.0, min(average_confidence or 0.0, 1.0))

    # Weighted blend: keyword presence and Arabic density matter most for
    # deciding whether OCR actually captured legal Arabic content.
    score = (
        arabic_ratio * 0.35
        + keyword_score * 0.30
        + line_score * 0.15
        + confidence_score * 0.20
    )

    if total_chars < MIN_USEFUL_TEXT_LENGTH:
        score *= 0.5

    return round(max(0.0, min(score, 1.0)), 3)
