"""
app.py
------
MediGuide AI — Retrieval-Augmented Healthcare Knowledge Assistant

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import datetime

import streamlit as st

from utils.pdf_loader import extract_text_from_multiple_pdfs
from utils.chunker import chunk_documents
from utils.embedding import embed_texts, embed_query
from utils.pinecone_db import get_pinecone_client, get_or_create_index, upsert_chunks, query_top_k, clear_index
from utils.rag import generate_answer, unique_sources

UPLOAD_DIR = "data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="MediGuide AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# STYLE — clinical, calm, editorial. Deep teal / ink on warm parchment.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #1c2b2a;
    --teal: #0f5257;
    --teal-deep: #0a3d40;
    --parchment: #faf7f0;
    --paper: #ffffff;
    --line: #e4ddc9;
    --gold: #b08d3e;
    --muted: #6b6355;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--parchment); }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Header banner */
.mg-header {
    background: linear-gradient(135deg, var(--teal-deep), var(--teal));
    padding: 2.2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 24px rgba(10,61,64,0.18);
}
.mg-header h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: #faf7f0;
    margin: 0;
    letter-spacing: 0.3px;
}
.mg-header p {
    color: #cfe3e0;
    font-size: 0.95rem;
    margin-top: 0.35rem;
    font-weight: 400;
}

/* Section cards */
.mg-card {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(28,43,42,0.04);
}
.mg-card h3 {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    color: var(--teal-deep);
    margin-top: 0;
    margin-bottom: 0.6rem;
}

/* Answer block */
.mg-answer {
    background: var(--paper);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    font-size: 0.98rem;
    line-height: 1.6;
}

/* Source pill */
.mg-source {
    display: inline-block;
    background: #eef3ee;
    color: var(--teal-deep);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 0.28rem 0.85rem;
    font-size: 0.8rem;
    margin: 0.2rem 0.35rem 0.2rem 0;
}

/* Chat history entries */
.mg-qa {
    padding: 0.9rem 0;
    border-bottom: 1px solid var(--line);
}
.mg-q {
    font-weight: 600;
    color: var(--teal-deep);
    font-size: 0.92rem;
    margin-bottom: 0.25rem;
}
.mg-time {
    color: var(--muted);
    font-size: 0.75rem;
    float: right;
}

/* Disclaimer */
.mg-disclaimer {
    background: #fbf3e3;
    border: 1px solid #e8d9ad;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #6b5626;
    margin-top: 2rem;
}

/* Buttons */
.stButton > button {
    background: var(--teal);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.1rem;
    font-weight: 500;
    transition: background 0.15s ease;
}
.stButton > button:hover { background: var(--teal-deep); color: white; }

/* Status pill */
.mg-status-ready { color: #1a7a3c; font-weight: 500; }
.mg-status-empty { color: var(--muted); font-weight: 500; }

/* Force sidebar to always stay open — prevents it from getting stuck
   collapsed with no visible arrow to reopen it. */
[data-testid="stSidebar"] {
    min-width: 300px !important;
    max-width: 300px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "processed_doc_names" not in st.session_state:
    st.session_state.processed_doc_names = []

# ----------------------------------------------------------------------
# SIDEBAR — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Check if keys exist in environment/secrets first, otherwise default to empty string
    default_pinecone = st.secrets.get("PINECONE_API_KEY", "")
    default_groq = st.secrets.get("GROQ_API_KEY", "")
    
    # If secrets are missing, let the user input them manually
    pinecone_api_key = st.text_input(
        "Pinecone API Key", 
        value=default_pinecone, 
        type="password",
        disabled=bool(default_pinecone)
    )
    groq_api_key = st.text_input(
        "Groq API Key", 
        value=default_groq, 
        type="password",
        disabled=bool(default_groq)
    )
    
    if default_pinecone and default_groq:
        st.caption("🔒 Using administrator API keys.")

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="mg-header">
    <h1>🩺 MediGuide AI</h1>
    <p>A Retrieval-Augmented Healthcare Knowledge Assistant — ask questions,
    get answers grounded in your own medical documents.</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.4])

# ----------------------------------------------------------------------
# LEFT COLUMN — document upload & processing
# ----------------------------------------------------------------------
with col_left:
    st.markdown('<div class="mg-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload trusted medical PDFs (WHO guidelines, first aid manuals, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("⚙️ Process Documents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not pinecone_api_key or not groq_api_key:
            st.warning("Please enter your Pinecone and Groq API keys in the sidebar.")
        else:
            with st.spinner("Reading PDFs..."):
                file_paths = {}
                for f in uploaded_files:
                    path = os.path.join(UPLOAD_DIR, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                    file_paths[f.name] = path
                page_records = extract_text_from_multiple_pdfs(file_paths)

            with st.spinner("Splitting into chunks..."):
                chunks = chunk_documents(page_records)

            with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
                vectors = embed_texts([c["text"] for c in chunks])

            with st.spinner("Storing vectors in Pinecone..."):
                pc = get_pinecone_client(pinecone_api_key)
                index = get_or_create_index(pc)
                upsert_chunks(index, chunks, vectors)

            st.session_state.documents_processed = True
            st.session_state.processed_doc_names = list(file_paths.keys())
            st.success(f"Indexed {len(chunks)} chunks from {len(file_paths)} document(s).")
            time.sleep(0.5)
            st.rerun()

    st.markdown('<div class="mg-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Chat History")
    if not st.session_state.chat_history:
        st.caption("Your questions and answers will appear here.")
    else:
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"""
            <div class="mg-qa">
                <span class="mg-time">{entry['time']}</span>
                <div class="mg-q">Q: {entry['question']}</div>
                <div style="font-size:0.88rem; color:#3a3a3a;">{entry['answer'][:180]}{'...' if len(entry['answer']) > 180 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# RIGHT COLUMN — ask a question
# ----------------------------------------------------------------------
with col_right:
    st.markdown('<div class="mg-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Ask a Question")
    question = st.text_input(
        "e.g. What are the symptoms of dengue fever?",
        label_visibility="collapsed",
        placeholder="e.g. What are the symptoms of dengue fever?",
    )
    ask_clicked = st.button("Ask", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please type a question.")
        elif not st.session_state.documents_processed:
            st.warning("Please upload and process documents first.")
        elif not pinecone_api_key or not groq_api_key:
            st.warning("Please enter your Pinecone and Groq API keys in the sidebar.")
        else:
            with st.spinner("Searching documents..."):
                pc = get_pinecone_client(pinecone_api_key)
                index = get_or_create_index(pc)
                q_vector = embed_query(question)
                matches = query_top_k(index, q_vector, k=top_k)

            with st.spinner("Generating answer..."):
                answer = generate_answer(groq_api_key, question, matches)
                sources = unique_sources(matches)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "time": datetime.now().strftime("%H:%M"),
            })

            st.markdown("#### Answer")
            st.markdown(f'<div class="mg-answer">{answer}</div>', unsafe_allow_html=True)

            if sources:
                st.markdown("#### Sources")
                for s in sources:
                    st.markdown(
                        f'<span class="mg-source">📄 {s["source"]} — Page {s["page"]} '
                        f'(match {s["score"]:.2f})</span>',
                        unsafe_allow_html=True,
                    )

# ----------------------------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------------------------
st.markdown("""
<div class="mg-disclaimer">
⚠ This application is intended for educational purposes only.
It should not be used for diagnosis or treatment.
Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)