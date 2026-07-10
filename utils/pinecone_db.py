"""
pinecone_db.py
--------------
Handles all interaction with Pinecone: creating the index, upserting
chunk vectors with metadata, and querying for the top-K most similar
chunks to a user's question.
"""

import time
from pinecone import Pinecone, ServerlessSpec

from utils.embedding import EMBEDDING_DIM

INDEX_NAME = "mediguide-ai"


def get_pinecone_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


def get_or_create_index(pc: Pinecone, index_name: str = INDEX_NAME):
    """Creates the index on first run, then just connects to it afterwards."""
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for the index to be ready before writing to it
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)


def upsert_chunks(index, chunks: list[dict], vectors: list[list[float]], batch_size: int = 100):
    """
    Writes chunk vectors + metadata to Pinecone in batches.

    chunks[i] must correspond to vectors[i].
    """
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_number": chunk["chunk_number"],
            },
        })

    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i:i + batch_size])


def query_top_k(index, query_vector: list[float], k: int = 5) -> list[dict]:
    """
    Searches Pinecone for the k chunks most semantically similar to the
    query vector. Returns metadata (text, source, page) plus a similarity
    score for each match.
    """
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    matches = []
    for match in results["matches"]:
        matches.append({
            "score": match["score"],
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "page": match["metadata"]["page"],
        })
    return matches


def clear_index(index):
    """Wipes all vectors — used by the 'Clear Chat / Reset Documents' action."""
    index.delete(delete_all=True)
