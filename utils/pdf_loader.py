"""
pdf_loader.py
-------------
Extracts text from uploaded PDF files, page by page, so that page
numbers can later be attached to chunks as metadata (used for
source citation in the UI).
"""

import pdfplumber


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF and returns a list of page-level records.

    Returns:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."},
            ...
        ]
    """
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:  # skip blank pages
                pages.append({"page": i, "text": text})
    return pages


def extract_text_from_multiple_pdfs(file_paths: dict[str, str]) -> list[dict]:
    """
    Args:
        file_paths: {document_name: path_on_disk}

    Returns:
        A flat list of page records, each tagged with its source document:
        [
            {"source": "WHO Guidelines.pdf", "page": 1, "text": "..."},
            ...
        ]
    """
    all_pages = []
    for doc_name, path in file_paths.items():
        pages = extract_text_from_pdf(path)
        for p in pages:
            all_pages.append({
                "source": doc_name,
                "page": p["page"],
                "text": p["text"],
            })
    return all_pages
