"""RAG adapter: loads repo embeddings from SQLite into a rag-pipeline FAISS index.

Requires rag-pipeline[google] installed:
    uv add --editable "../rag_pipeline[google]"
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.embed import build_text_for_embedding
from pipeline.store import DB_PATH, get_connection
from rag_pipeline import GeneratorResponse, build_index_from_vectors, generate, retrieve

_SYSTEM_PROMPT = """You are GovScan Intelligence, an AI analyst specialising in government open-source technology.
Answer questions using only the repository information provided in the context below.
Each entry is a government repository — cite sources using their full id (e.g. alphagov/notify).
If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}"""


@st.cache_resource(show_spinner="Building search index…")
def _get_index():
    """Load SQLite repo embeddings into an in-memory FAISS index.

    Cached for the Streamlit session lifetime — takes ~1s for a few thousand
    repos and requires no API calls (vectors already exist in the DB).
    """
    with get_connection(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT id, name, country, domain, llm_summary, topics, description, embedding
            FROM repos
            WHERE embedding IS NOT NULL
        """).fetchall()

    texts, vectors, metadatas = [], [], []
    for r in rows:
        texts.append(build_text_for_embedding(dict(r)))
        vectors.append(np.frombuffer(r["embedding"], dtype=np.float32).tolist())
        metadatas.append({
            "source": r["id"],       # "org/repo-name" — used as citation
            "name": r["name"],
            "country": r["country"],
            "domain": r["domain"],
        })

    return build_index_from_vectors(texts, vectors, metadatas)


def ask_rag(query: str, api_key: str) -> GeneratorResponse:
    """Retrieve relevant repos and generate a grounded answer.

    Returns GeneratorResponse with:
        .answer  — markdown string
        .sources — list of "org/repo-name" strings (deduplicated, ordered by relevance)
    """
    store = _get_index()
    docs = retrieve(store, query)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    return generate(query, docs, system_prompt=_SYSTEM_PROMPT, llm=llm)
