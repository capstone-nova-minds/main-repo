"""Thin wrapper around requests calls to the FastAPI backend.

API_BASE_URL is the single source of truth for the backend address --
every Streamlit page must import it (or the functions below) from here
instead of hardcoding a URL.

- Docker Compose sets API_BASE_URL=http://backend:8000 explicitly (see
  docker-compose.yml), since "backend" is only resolvable on the
  Docker Compose network.
- Running locally (no Docker), API_BASE_URL is unset, so it falls back
  to http://127.0.0.1:8000, matching `uvicorn main:app --port 8000`
  run directly on the host.
"""

import os
from typing import Tuple

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

DEFAULT_TIMEOUT = 120  # OCR/NER can take a while on CPU
HEALTH_TIMEOUT = 5  # health checks should fail fast, not hang for 2 minutes

# /process runs the full pipeline (OCR engine attempts across every page,
# then local NER) on CPU. The very first call after the backend starts
# also pays a one-time cost to load the EasyOCR and Stanza Arabic models
# (and, until they fail once, PaddleOCR's) -- comfortably longer than the
# general-purpose DEFAULT_TIMEOUT above.
PROCESS_TIMEOUT = int(os.getenv("PROCESS_TIMEOUT", "600"))


def check_backend_health() -> Tuple[bool, str]:
    """GET {API_BASE_URL}/healthz. Returns (is_healthy, message)."""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=HEALTH_TIMEOUT)
        response.raise_for_status()
        return True, "Backend is reachable."
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def upload_document(file):
    files = {"file": (file.name, file.getvalue())}
    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def process_document(document_id: str):
    response = requests.post(f"{API_BASE_URL}/process/{document_id}", timeout=PROCESS_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_result(document_id: str):
    response = requests.get(f"{API_BASE_URL}/result/{document_id}", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_review(document_id: str, reviewed_data: dict):
    response = requests.post(
        f"{API_BASE_URL}/review/{document_id}", json=reviewed_data, timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def export_json_url(document_id: str) -> str:
    return f"{API_BASE_URL}/export/json/{document_id}"


def export_excel_url(document_id: str) -> str:
    return f"{API_BASE_URL}/export/excel/{document_id}"


def download_export(url: str) -> bytes:
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.content
