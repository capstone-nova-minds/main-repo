import re
from typing import Any

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"


def get_complete_line(text: str, start: int, end: int) -> str:
    """Return the full line containing the detected entity."""
    line_start = text.rfind("\n", 0, start)

    if line_start == -1:
        line_start = 0
    else:
        line_start += 1

    line_end = text.find("\n", end)

    if line_end == -1:
        line_end = len(text)

    return text[line_start:line_end].strip()


def clean_company_name(line: str) -> str:
    """Extract the full company name from its OCR line."""
    line = re.sub(r"^[\s\-•–—]+", "", line).strip()

    company_position = line.find("شركة")

    if company_position == -1:
        return ""

    company_text = line[company_position:]

    company_text = re.split(
        r"\s*(?:"
        r"\||"
        r"رقم\s*التسجيل|"
        r"الرقم\s*الوطني|"
        r"نوع\s*الشخص|"
        r"REG\s*-?\s*\d+"
        r")",
        company_text,
        maxsplit=1,
    )[0]

    company_text = re.sub(r"\s+", " ", company_text)

    return company_text.strip(" :-|")


def postprocess_entity(
    text: str,
    entity: dict[str, Any],
) -> dict[str, Any]:
    """Post-process CAMeLBERT output."""
    start = int(entity.get("start", 0))
    end = int(entity.get("end", 0))
    label = str(entity.get("entity_group", ""))
    score = float(entity.get("score", 0))

    entity_text = text[start:end].strip()
    extraction_method = "camelbert_exact_span"

    if label == "ORG":
        full_line = get_complete_line(
            text=text,
            start=start,
            end=end,
        )

        expanded_company = clean_company_name(full_line)

        if expanded_company:
            entity_text = expanded_company
            extraction_method = "camelbert_org_line_expansion"

    return {
        "text": entity_text,
        "label": label,
        "confidence": round(score, 4),
        "start": start,
        "end": end,
        "extraction_method": extraction_method,
    }


def main() -> None:
    print("Loading CAMeLBERT NER model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME
    )

    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )

    text = """
الأشخاص / الجهات المطلوبة:

يوسف سامر فؤاد العزام
الرقم الوطني: 98100000001
نوع الشخص: فرد

ليث محمود سليم الخطيب
الرقم الوطني: 98100000002
نوع الشخص: فرد

شركة الريادة للخدمات المالية
رقم التسجيل: REG-202701
نوع الشخص: شركة
"""

    raw_entities = ner(text)

    print("\nDetected entities:")

    for raw_entity in raw_entities:
        processed = postprocess_entity(
            text=text,
            entity=raw_entity,
        )
        print(processed)


if __name__ == "__main__":
    main()
