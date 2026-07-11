"""UK AI Policy Assistant — RAG over UK government AI strategy PDFs (2021–2024).

Uses vector search via rag-pipeline + a pre-built FAISS index (data/uk_ai_policy_faiss/).
Requires fastembed/onnxruntime to be installed to embed the query at search time —
if that's unavailable in this environment (see pipeline/rag.py for the same caveat
on Streamlit Cloud), this page shows an error rather than falling back silently.
"""
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="UK AI Policy — GovScan", page_icon="📜", layout="centered")

st.title("📜 UK AI Policy Assistant")
st.caption(
    "Ask questions about UK government AI strategy documents (2021–2024). "
    "Answers are grounded in the source PDFs, with citations."
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set.")
    st.stop()

INDEX_PATH = Path(__file__).parent.parent / "data" / "uk_ai_policy_faiss"

try:
    from rag_pipeline import generate as rag_generate
    from rag_pipeline import load_index, retrieve
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    st.error(
        "Vector search isn't available in this environment — rag-pipeline's "
        "embedding stack (fastembed/onnxruntime) failed to install."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading policy document index…")
def _get_index():
    return load_index(INDEX_PATH)


_SYSTEM_PROMPT = """You are a policy analyst assistant with access to UK government \
AI strategy documents.

Rules:
- Only use information from the provided context passages
- If the context does not contain the answer, say: "The documents provided do not \
address this question directly."
- Always cite your sources: name the document and page number
- Be concise and precise — you are writing for a policy audience, not a general one
- Never speculate beyond what the documents say

Context:
{context}"""

examples = [
    "What are the three pillars of the UK National AI Strategy?",
    "What are the five cross-sector principles in the AI regulation white paper?",
    "What was the Bletchley Declaration?",
    "How does the UK define frontier AI?",
]

if "uk_ai_messages" not in st.session_state:
    st.session_state.uk_ai_messages = []

for msg in st.session_state.uk_ai_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.uk_ai_messages:
    st.markdown("**Example questions:**")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True, key=f"uk_ai_ex_{i}"):
            st.session_state.uk_ai_pending = example
            st.rerun()

prompt = st.chat_input("Ask about UK AI policy...")

if not prompt and "uk_ai_pending" in st.session_state:
    prompt = st.session_state.pop("uk_ai_pending")

if prompt:
    st.session_state.uk_ai_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            store = _get_index()
            docs = retrieve(store, prompt)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", google_api_key=api_key, temperature=0
            )
            response = rag_generate(
                prompt,
                docs,
                system_prompt=_SYSTEM_PROMPT,
                llm=llm,
            )
        st.markdown(response.answer)
        if docs:
            with st.expander(f"Sources — {len(docs)} passages"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**[{i}] {source} — page {page}**")
                    st.text(doc.page_content)
        answer = response.answer

    st.session_state.uk_ai_messages.append({"role": "assistant", "content": answer})
