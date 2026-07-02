import os

import streamlit as st

st.set_page_config(page_title="Ask GovScan", page_icon="💬", layout="centered")

st.title("Ask GovScan")
st.caption("Ask anything about government open-source technology across 16 countries.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set.")
    st.stop()

mode = st.radio(
    "Answer mode",
    ["SQL — structured query", "RAG — semantic search"],
    horizontal=True,
    help=(
        "**SQL** — precise counts, filters, and comparisons across the full dataset.\n\n"
        "**RAG** — exploratory questions about what governments are building; "
        "answers grounded in repo descriptions with source citations."
    ),
)

# Reset conversation when the user switches modes
if st.session_state.get("_mode") != mode:
    st.session_state.messages = []
    st.session_state["_mode"] = mode

if mode.startswith("SQL"):
    try:
        from pipeline.query_agent import QueryAgent
        agent = QueryAgent(api_key=api_key)
    except Exception as exc:
        st.error(f"Could not load SQL agent: {exc}")
        st.stop()
    examples = [
        "Which countries have the most AI/ML repositories?",
        "What are the top languages used by UK government orgs?",
        "Which governments use open-weight LLMs like Llama or Mistral?",
        "Find clusters where multiple countries built similar tools independently.",
    ]
else:
    from pipeline.rag import ask_rag
    examples = [
        "What are governments building to help citizens access benefits?",
        "Which repos focus on transparency or open data publishing?",
        "What health technology projects exist across different countries?",
        "Show me examples of government projects using large language models.",
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown("**Example questions:**")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

prompt = st.chat_input("Ask about government tech trends...")

if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if mode.startswith("SQL"):
            with st.spinner("Querying database..."):
                answer = agent.ask(prompt, st.session_state.messages[:-1])
            st.markdown(answer)
        else:
            with st.spinner("Searching repos and generating answer..."):
                response = ask_rag(prompt, api_key)
            st.markdown(response.answer)
            if response.sources:
                with st.expander(f"Sources — {len(response.sources)} repos"):
                    for src in response.sources:
                        st.markdown(f"- [{src}](https://github.com/{src})")
            answer = response.answer

    st.session_state.messages.append({"role": "assistant", "content": answer})
