import streamlit as st

from pipeline.store import get_all_repos, get_stats

st.set_page_config(
    page_title="GovScan",
    page_icon="🏛",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_stats():
    return get_stats()


@st.cache_data(ttl=3600)
def load_repos():
    return get_all_repos()


stats = load_stats()
repos = load_repos()
classified = [r for r in repos if r["domain"]]

st.title("🏛 GovScan")
st.markdown("Tracking open-source work across government GitHub organisations worldwide.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total repos", f"{stats['total_repos']:,}")
c2.metric("Countries", stats["countries"])
c3.metric("AI / ML repos", stats["ai_ml_repos"])
c4.metric("Duplicate efforts", stats["clusters"])

if stats["last_updated"]:
    st.caption(f"Pipeline last ran: {stats['last_updated'][:10]}")

if stats["total_repos"] > 0:
    pct = len(classified) / stats["total_repos"] * 100
    st.progress(
        pct / 100,
        text=f"{len(classified):,} of {stats['total_repos']:,} repos classified ({pct:.0f}%)",
    )

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/1_overview.py", label="**📊 Overview**")
    st.caption("Domain breakdown and repos by country")
with col2:
    st.page_link("pages/2_trends.py", label="**📈 Trends**")
    st.caption("Activity over time and language breakdown")
with col3:
    st.page_link("pages/3_similarity.py", label="**🔁 Similarity**")
    st.caption("Clusters of similar work across governments")
with col4:
    st.page_link("pages/4_search.py", label="**🔍 Search**")
    st.caption("Browse and filter all classified repos")
