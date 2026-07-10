# AI Court Order Information Extraction System

A fully local system that extracts structured information from Arabic
Jordanian court attachment orders, lets a human review and correct the
result, and exports the approved data to JSON and Excel.

## Local-only, no LLM, no cloud AI

- **No LLM** of any kind is used (no GPT, Gemini, Claude API, OpenAI API, or
  any other large language model).
- **No cloud AI API** and **no cloud OCR** are used.
- Every OCR, NER, and extraction step runs **locally** inside Docker
  containers on your machine. No document ever leaves your computer.

## Technology Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, requests |
| PDF processing | PyMuPDF |
| Image processing | OpenCV, Pillow, NumPy |
| OCR | EasyOCR (primary), Tesseract (fallback), PaddleOCR (future stub) |
| NER | Stanza Arabic NER (primary), CAMeL Tools (future stub) |
| Extraction | Regex, Arabic legal keyword rules, National ID validation |
| Export | pandas, openpyxl |
| Containers | Docker, docker-compose |

## Running the project

```bash
docker compose up --build
```

- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Ports 8000 (backend) and 8501 (frontend) are fixed and must not change.

## Folder Structure

```
backend/            FastAPI app: api/, services/ (incl. ocr_engines/, ner_engines/), schemas/, utils/
streamlit_app/       Streamlit app: pages/, components/, utils/
data/                Shared volume: uploads, pages, processed, ocr_outputs, extracted, reviewed, exports
tests/               Unit tests + sample documents / expected outputs
docs/                git_workflow.md, technical_documentation.md, user_guide.md, test_results.md
docker-compose.yml
.env.example
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Health check |
| POST | `/upload` | Upload a PDF/JPG/JPEG/PNG court order |
| POST | `/process/{document_id}` | Run the full OCR + NER + extraction pipeline |
| GET | `/result/{document_id}` | Fetch the extracted result |
| POST | `/review/{document_id}` | Save human-reviewed/edited result |
| GET | `/export/json/{document_id}` | Download JSON (reviewed data if available) |
| GET | `/export/excel/{document_id}` | Download Excel (reviewed data if available) |

## Pipeline

```
Upload -> PDF/Image processing -> Image preprocessing -> OCR Router/Fallback
-> Text normalization -> Rule-based document extraction -> Rule-based person
extraction -> Local Arabic NER -> Merge rules + NER -> Validation + confidence
-> Streamlit human review -> Export JSON / Excel
```

Human review (the Review page, `POST /review/{document_id}`) is mandatory
before export in the intended workflow: if no reviewed data exists yet,
export falls back to the raw extracted data.

## Git Workflow

See [docs/git_workflow.md](docs/git_workflow.md) for full details.
Summary: `main` is production-ready, `develop` is the integration branch,
each `feature/*` branch owns one component, and all work merges into
`develop` via pull request -- never push directly to `main`.

## Limitations

- OCR and NER accuracy depend on scan quality; low-quality scans will be
  flagged for review but may still contain errors.
- Grouped Arabic name splitting only handles clearly recognizable
  "first names + family tail" patterns; ambiguous cases are left unsplit
  and flagged `needs_review`.
- PaddleOCR and CAMeL Tools are stubbed for future use, not implemented.
- Stanza's Arabic model must be downloaded (attempted at Docker build
  time); if unavailable, NER is skipped and the system falls back to
  rule-based extraction only.




# one-time: install the missing backend deps into the existing venv

.\.venv\Scripts\pip install fastapi opencv-python-headless easyocr pytesseract stanza



# terminal 1 — backend

cd backend

..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000



# terminal 2 — frontend

cd streamlit_app

$env:API_BASE_URL python -m streamlit run app.py= "http://localhost:8000"   # default is http://backend:8000, only valid inside Docker