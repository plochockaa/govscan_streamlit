import os

import streamlit as st

st.set_page_config(page_title="Ask GovScan", page_icon="💬", layout="centered")

st.title("Ask GovScan")
st.caption("Ask anything about government open-source technology across 16 countries.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set.")
    st.stop()

try:
    from pipeline.query_agent import QueryAgent
    agent = QueryAgent(api_key=api_key)
except Exception as exc:
    st.error(f"Could not load agent: {exc}")
    st.stop()

EXAMPLES = [
    "Which countries have the most AI/ML repositories?",
    "What are the top languages used by UK government orgs?",
    "Which governments use open-weight LLMs like Llama or Mistral?",
    "Find clusters where multiple countries built similar tools independently.",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown("**Example questions:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
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
        with st.spinner("Querying database..."):
            answer = agent.ask(prompt, st.session_state.messages[:-1])
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
