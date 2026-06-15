import json

import pandas as pd
import streamlit as st

from pipeline.store import get_all_repos

st.set_page_config(page_title="Search — GovScan", layout="wide")

DOMAIN_LABELS = {
    "ai_ml":               "AI / ML",
    "data_infrastructure": "Data Infrastructure",
    "citizen_services":    "Citizen Services",
    "security":            "Security",
    "open_data":           "Open Data",
    "devtools":            "Developer Tools",
    "research":            "Research",
    "policy_tools":        "Policy Tools",
    "other":               "Other",
}


@st.cache_data(ttl=3600)
def load():
    repos = get_all_repos()
    classified = [r for r in repos if r["domain"]]
    df = pd.DataFrame(classified)
    df["topics_list"] = df["topics"].apply(
        lambda t: json.loads(t) if isinstance(t, str) else (t or [])
    )
    df["github_url"] = "https://github.com/" + df["id"]
    df["domain_label"] = df["domain"].map(DOMAIN_LABELS).fillna("Other")
    return df


st.title("🔍 Search")

df = load()

if df.empty:
    st.info("No classified repos yet — the pipeline is still running.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter")

    query = st.text_input("Search name / description / summary", placeholder="e.g. payments, accessibility")

    countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries)

    domains = ["All"] + sorted(df["domain_label"].dropna().unique().tolist())
    sel_domain = st.selectbox("Domain", domains)

    languages = ["All"] + sorted(df["language"].dropna().unique().tolist())
    sel_language = st.selectbox("Language", languages)

    maturities = ["All"] + sorted(df["maturity"].dropna().unique().tolist())
    sel_maturity = st.selectbox("Maturity", maturities)

    min_stars = st.number_input("Min stars", min_value=0, value=0, step=10)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()

if query:
    q = query.lower()
    mask = (
        filtered["name"].str.lower().str.contains(q, na=False)
        | filtered["description"].str.lower().str.contains(q, na=False)
        | filtered["llm_summary"].str.lower().str.contains(q, na=False)
    )
    filtered = filtered[mask]

if sel_country != "All":
    filtered = filtered[filtered["country"] == sel_country]
if sel_domain != "All":
    filtered = filtered[filtered["domain_label"] == sel_domain]
if sel_language != "All":
    filtered = filtered[filtered["language"] == sel_language]
if sel_maturity != "All":
    filtered = filtered[filtered["maturity"] == sel_maturity]
if min_stars > 0:
    filtered = filtered[filtered["stars"] >= min_stars]

filtered = filtered.sort_values("stars", ascending=False)

st.caption(f"{len(filtered)} repos match")

# ── Results ───────────────────────────────────────────────────────────────────
if filtered.empty:
    st.warning("No repos match the current filters.")
    st.stop()

for _, row in filtered.iterrows():
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"**[{row['name']}]({row['github_url']})** "
                f"`{row['org']}` · {row['country']}"
            )
            if row.get("llm_summary"):
                st.caption(row["llm_summary"])
            elif row.get("description"):
                st.caption(row["description"])
        with col2:
            st.markdown(
                f"⭐ {row['stars']:,}  \n"
                f"`{row['domain_label']}`  \n"
                f"`{row['maturity']}`"
            )
        st.divider()
