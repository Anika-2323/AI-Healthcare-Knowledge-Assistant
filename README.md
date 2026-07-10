# 🩺 MediGuide AI
### A Retrieval-Augmented Healthcare Knowledge Assistant

MediGuide AI lets you upload trusted medical PDFs (WHO guidelines, first-aid
manuals, public health handbooks) and ask questions in plain English. Instead
of scrolling through pages of documentation, the app retrieves the most
relevant passages and generates an answer grounded in that source material —
with the exact document and page number cited.

---

## How it works

```
Upload PDFs → Extract text → Chunk → Embed → Store in Pinecone
                                                     │
User question → Embed question → Search Pinecone ◄──┘
                                        │
                              Top-K relevant chunks
                                        │
                        Prompt LLM (context + question)
                                        │
                              Answer + cited sources
```

- **Chunking**: text is split into ~300-word chunks with 50-word overlap, so
  meaning isn't lost at chunk boundaries.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — a free, local
  model that turns each chunk into a 384-dimensional vector.
- **Vector store**: Pinecone serverless index (`mediguide-ai`), storing the
  vector plus metadata (source document, page, chunk number).
- **Retrieval**: cosine similarity search returns the top-K most relevant
  chunks for a given question.
- **Generation**: Groq's Llama 3.1 model answers strictly from the retrieved
  context. If the answer isn't in the documents, it says so explicitly rather
  than hallucinating.

---

## Setup

```bash
cd MediGuideAI
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need two free API keys, entered directly in the app's sidebar (never
hardcoded):

- **Pinecone**: [app.pinecone.io](https://app.pinecone.io) → free tier is enough for a demo.
- **Groq**: [console.groq.com](https://console.groq.com) → free tier, fast inference.

Run the app:

```bash
streamlit run app.py
```

---

## Suggested demo documents

Keep it to 3–5 PDFs for a clean demo:
- WHO disease guidance documents (public domain)
- First-aid manuals
- Public disease-prevention guides

---

## Talking points for interviews

- **What is RAG and why use it?** An LLM alone can hallucinate or rely on
  stale training data. RAG grounds every answer in retrieved, verifiable
  source text — the model only elaborates on what it's given.
- **Why chunk instead of embedding whole documents?** A single vector for an
  entire PDF would be too coarse to retrieve accurately; smaller chunks let
  semantic search zero in on the specific passage that answers the question.
- **Why a vector database?** Pinecone lets you search by *meaning* (cosine
  similarity between embeddings) instead of exact keyword matching — so
  "symptoms of dengue" can match a passage that never uses the word "symptom."
- **Why show sources?** Citing the document and page number lets a user
  verify the answer themselves, which matters enormously in a healthcare
  context where trust and accuracy are non-negotiable.
- **Why constrain the prompt to "answer only from context"?** It's the main
  lever for reducing hallucination — the model is explicitly told to decline
  rather than guess when the retrieved chunks don't contain the answer.

---

## Disclaimer

This application is intended for educational purposes only. It should not be
used for diagnosis or treatment. Always consult a qualified healthcare
professional.
