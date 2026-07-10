"""
rag.py
------
Ties retrieval and generation together:
  1. Build a grounded prompt from the retrieved chunks.
  2. Call the LLM (Groq / Llama) with that prompt.
  3. Return the answer alongside the sources used, for display + citation.
"""

from groq import Groq

LLM_MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a healthcare knowledge assistant.
Answer ONLY using the provided context from the uploaded documents.
Do not use outside knowledge, and do not guess.
If the answer is not available in the context, reply exactly:
"I couldn't find this information in the uploaded documents."

Keep answers clear, concise, and clinically accurate to the source text.
Do not provide medical advice, diagnosis, or treatment recommendations —
only summarize what the documents say."""


def build_context_block(matches: list[dict]) -> str:
    """Formats retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for m in matches:
        blocks.append(f"[Source: {m['source']}, Page {m['page']}]\n{m['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(groq_api_key: str, question: str, matches: list[dict]) -> str:
    """
    Sends the question + retrieved context to the LLM and returns the
    generated answer text.
    """
    if not matches:
        return "I couldn't find this information in the uploaded documents."

    context = build_context_block(matches)

    user_prompt = f"""Context:
{context}

Question:
{question}"""

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def unique_sources(matches: list[dict]) -> list[dict]:
    """De-duplicates (source, page) pairs for a clean 'Sources' display."""
    seen = set()
    result = []
    for m in matches:
        key = (m["source"], m["page"])
        if key not in seen:
            seen.add(key)
            result.append({"source": m["source"], "page": m["page"], "score": m["score"]})
    return result
