"""
embedding.py
------------
Wraps sentence-transformers/all-MiniLM-L6-v2 to turn text into
384-dimensional vectors. This is a local, free model (no API cost),
which makes the demo cheap and fast to run.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Cached so the model loads into memory only once per session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a list of strings and returns a list of vectors."""
    model = load_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embeds a single user question."""
    model = load_embedding_model()
    vector = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return vector[0].tolist()
