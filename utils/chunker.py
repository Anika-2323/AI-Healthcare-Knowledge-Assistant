"""
chunker.py
----------
Splits page-level text into overlapping word-based chunks.

Why chunking matters (interview talking point):
  - Embedding models have a limited context window and produce a single
    fixed-size vector per input. A whole PDF crammed into one vector would
    be too vague to retrieve accurately.
  - Overlap prevents losing meaning at chunk boundaries (e.g. a sentence
    describing symptoms that starts in chunk 3 and finishes in chunk 4).
"""

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
                overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Splits a single string of text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step back to create overlap
    return chunks


def chunk_documents(page_records: list[dict]) -> list[dict]:
    """
    Takes page-level records (from pdf_loader) and produces chunk-level
    records ready for embedding.

    Input:
        [{"source": "WHO.pdf", "page": 1, "text": "..."}]

    Output:
        [
            {
                "id": "WHO.pdf-p1-c0",
                "source": "WHO.pdf",
                "page": 1,
                "chunk_number": 0,
                "text": "..."
            },
            ...
        ]
    """
    chunked = []
    for record in page_records:
        pieces = chunk_text(record["text"])
        for i, piece in enumerate(pieces):
            chunked.append({
                "id": f"{record['source']}-p{record['page']}-c{i}",
                "source": record["source"],
                "page": record["page"],
                "chunk_number": i,
                "text": piece,
            })
    return chunked
