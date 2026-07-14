FROM python:3.11-slim

WORKDIR /app

# System dependencies needed for OpenCV, EasyOCR, and Tesseract Arabic OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Try to pre-download the Stanza Arabic model at build time so the first
# request doesn't pay the download cost. If this fails (e.g. no network
# during build), the app falls back to rule-based extraction at runtime.
RUN python -c "import stanza; stanza.download('ar')" || echo "Stanza Arabic model download skipped (no network at build time)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
