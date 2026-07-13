# Technical Documentation

## System Architecture

```
Upload (Streamlit) -> FastAPI /upload -> data/uploads/{document_id}
                                          |
                                    POST /process/{document_id}
                                          v
        PDF/Image -> Preprocessing -> OCR Router -> Normalization
                                          v
                Rule-based document extraction
                Rule-based person/company extraction
                Local Arabic NER (Stanza)
                Entity merge (rules + NER)
                Name splitting, validation, confidence scoring
                                          v
                        data/extracted/{document_id}.json
                                          v
        Streamlit Review screen (GET /result, POST /review)
                                          v
                        data/reviewed/{document_id}.json
                                          v
        Export (GET /export/json, GET /export/excel)
```

Everything above runs inside two Docker containers (`backend`, `frontend`)
that share a single `./data` volume. No component makes any network call
outside the user's machine.

## OCR Router and Fallback

`backend/services/ocr_router_service.py` never calls a specific OCR engine
directly -- it always goes through the router:

1. Try **EasyOCR** (`services/ocr_engines/easyocr_engine.py`), the primary
   engine, on the preprocessed page image.
2. Score the result with `utils/ocr_quality.py::calculate_ocr_quality`,
   which blends Arabic character ratio, legal-keyword presence, line
   count, and the engine's own confidence into one 0-1 score.
3. If EasyOCR failed, returned no text, or scored below
   `OCR_QUALITY_THRESHOLD`, also try **Tesseract**
   (`services/ocr_engines/tesseract_engine.py`, `lang=ara+eng`).
4. Whichever engine produced the higher quality score wins for that page.
   Both attempts (including errors) are recorded under `engine_attempts`.
5. **PaddleOCR** (`services/ocr_engines/paddleocr_engine_stub.py`) is a
   placeholder only -- `is_available()` always returns `False`. It can be
   wired into the router later without changing the router's logic.

If neither engine succeeds, the page is marked `needs_review=true` with an
empty text result; the pipeline continues rather than crashing.

## Local Arabic NER Layer

`backend/services/ner_service.py` wraps the primary NER engine, **Stanza**
(`services/ner_engines/stanza_ner_engine.py`), which loads its Arabic
pipeline lazily and runs fully offline once the model is present (the
backend image tries to pre-download it at build time).

- If NER is disabled (`ENABLE_NER=false`), the model isn't available, or
  extraction throws, `run_ner()` returns `ner_status="failed"` and the
  pipeline **continues using rule-based extraction only** -- NER never
  blocks or crashes the app.
- `services/ner_engines/camel_tools_ner_engine_stub.py` is a placeholder
  for a possible second NER engine; `is_available()` is always `False`.

NER supports the rule-based extraction, it does not replace it: rule-based
candidates are computed first, and NER only adds names the keyword rules
missed, or confirms company/individual classification.

## Rule-Based Extraction

- **Document fields** (`document_extraction_service.py`): deterministic
  keyword + regex matching for court name, case number, document number,
  and document date. Every field returns `{value, confidence, needs_review}`.
- **Person/company candidates** (`person_extraction_service.py`): lines
  containing legal keywords (`المطلوب`, `المدين`, `شركة`, ...) are treated
  as candidates; an 11-digit National ID regex
  (`(?<!\d)\d{11}(?!\d)`) attaches IDs found on the same line; company
  keywords (`شركة`, `مؤسسة`, `سجل تجاري`, `رقم تسجيل`) flag `person_type=Company`.
- **Name splitting** (`name_splitting_service.py`): grouped family names
  (e.g. `أحمد ومحمد وخالد أبناء محمود سالم`) are split into individual
  records only when a clear "first names + family tail" pattern is found;
  otherwise the name is left unsplit and flagged `needs_review=true`.

## Entity Merging

`entity_merge_service.py` combines rule-based candidates with NER entities:

- NER `PERSON`/`PER` entities are added only if they aren't already a
  simple duplicate (exact match or substring match) of an existing
  candidate name.
- A National ID found within a small character window around a NER
  `PERSON` entity is attached to that new record; without one, the record
  is marked `needs_review=true`.
- NER `ORG`/`ORGANIZATION` entities become `Company` records; if the
  surrounding text contains company keywords, an existing rule-based
  record is also confirmed as a Company.

## Validation and Confidence

`validation_service.py` enforces the business rules: National IDs must be
`null` or exactly 11 digits, `person_type` must be `Individual` or
`Company` (companies never keep a National ID), and any field/record with
confidence below `CONFIDENCE_THRESHOLD` (default `0.70`) is flagged
`needs_review=true`. Confidence bands: `0.90-1.00` High, `0.70-0.89`
Medium, below `0.70` Needs Review.

## Export Logic

`export_service.py` always prefers `data/reviewed/{document_id}.json` over
`data/extracted/{document_id}.json` (see `review_service.get_best_available_result`).
JSON export returns the stored result as-is; Excel export flattens it to
one row per person, repeating the document-level fields on every row,
using `pandas` + `openpyxl`.

## Data Privacy Notes

- All processing happens locally inside the two Docker containers; no
  external API, cloud OCR, or LLM is ever called.
- Uploaded files and all intermediate artifacts stay under `./data`,
  which is git-ignored except for `.gitkeep` placeholders.
- National IDs are never logged or printed to the terminal -- only
  written to the JSON/Excel result files the user explicitly reviews and
  exports.
