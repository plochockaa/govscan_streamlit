"""RAG retrieval with Gemini generation.

Uses vector search when rag-pipeline + fastembed are available (local dev with --extra pipeline).
Falls back to keyword search on Streamlit Cloud where onnxruntime/fastembed is unavailable.
"""
from __future__ import annotations

import re
import requests
from dataclasses import dataclass, field

import streamlit as st

from pipeline.store import DB_PATH, get_connection

_MODEL = "gemini-2.5-flash"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_SYSTEM_PROMPT = """You are GovScan Intelligence, an AI analyst specialising in government open-source technology.
Answer questions using only the repository information provided in the context below.
Each entry is a government repository — cite sources using their full id (e.g. alphagov/notify).
If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}"""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[str] = field(default_factory=list)


try:
    from rag_pipeline import build_index_from_vectors as _build_index
    from rag_pipeline import retrieve as _retrieve
    from rag_pipeline import generate as _rag_generate
    from langchain_google_genai import ChatGoogleGenerativeAI
    _VECTOR_SEARCH = True
except ImportError:
    _VECTOR_SEARCH = False


if _VECTOR_SEARCH:
    @st.cache_resource(show_spinner="Building search index…")
    def _get_index():
        import numpy as np
        from pipeline.embed import build_text_for_embedding

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
                "source": r["id"],
                "name": r["name"],
                "country": r["country"],
                "domain": r["domain"],
            })
        return _build_index(texts, vectors, metadatas)


def _text_retrieve(query: str, top_k: int = 15) -> list[dict]:
    """Keyword search against llm_summary and description in SQLite."""
    tokens = [t.lower() for t in re.findall(r'\w+', query) if len(t) > 2]
    if not tokens:
        return []

    with get_connection(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT id, country, domain, llm_summary, description
            FROM repos
            WHERE llm_summary IS NOT NULL
        """).fetchall()

    scored = []
    for r in rows:
        text = (
            f"{r['llm_summary'] or ''} {r['description'] or ''} "
            f"{r['domain'] or ''} {r['country'] or ''}"
        ).lower()
        score = sum(text.count(t) for t in tokens)
        if score > 0:
            scored.append((score, dict(r)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def _generate(query: str, docs: list[dict], api_key: str) -> GeneratorResponse:
    """Generate a grounded answer from retrieved docs via Gemini HTTP API."""
    context = "\n\n".join(
        f"[{r['id']}] ({r.get('country', '')}, {r.get('domain', '')}): "
        f"{r.get('llm_summary') or r.get('description', '')}"
        for r in docs
    )
    sources = list(dict.fromkeys(r["id"] for r in docs))
    prompt = _SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {query}"

    resp = requests.post(
        f"{_API_BASE}/models/{_MODEL}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
    answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return GeneratorResponse(answer=answer, sources=sources)


def ask_rag(query: str, api_key: str) -> GeneratorResponse:
    """Retrieve relevant repos and generate a grounded answer."""
    if _VECTOR_SEARCH:
        store = _get_index()
        docs = _retrieve(store, query)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
        resp = _rag_generate(query, docs, system_prompt=_SYSTEM_PROMPT, llm=llm)
        return GeneratorResponse(answer=resp.answer, sources=list(resp.sources))

    docs = _text_retrieve(query)
    return _generate(query, docs, api_key)
