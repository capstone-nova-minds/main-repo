"""Thin wrapper around requests calls to the FastAPI backend."""

import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

DEFAULT_TIMEOUT = 120  # OCR/NER can take a while on CPU


def upload_document(file):
    files = {"file": (file.name, file.getvalue())}
    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def process_document(document_id: str):
    response = requests.post(f"{API_BASE_URL}/process/{document_id}", timeout=DEFAULT_TIMEOUT)
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
