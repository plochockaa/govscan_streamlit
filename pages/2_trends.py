import json

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.detect import FRAMEWORKS, FRONTIER, OPEN_WEIGHT
from pipeline.store import get_all_repos

st.set_page_config(page_title="Trends — GovScan", layout="wide")


@st.cache_data(ttl=3600)
def load():
    return get_all_repos()


st.title("📈 Trends")

repos = load()
if not repos:
    st.info("No data yet.")
    st.stop()

df = pd.DataFrame(repos)
df["created_year"] = pd.to_datetime(df["created_at"], errors="coerce").dt.year
df["updated_year"] = pd.to_datetime(df["updated_at"], errors="coerce").dt.year

# ── Row 1: repos created per year + top languages ────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Repos created per year")
    year_counts = (
        df["created_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    year_counts.columns = ["year", "count"]
    fig = px.line(
        year_counts, x="year", y="count",
        markers=True,
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="New repos", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top languages")
    lang_counts = (
        df["language"]
        .dropna()
        .value_counts()
        .head(15)
        .reset_index()
    )
    lang_counts.columns = ["language", "count"]
    fig2 = px.bar(
        lang_counts, x="count", y="language",
        orientation="h",
        color_discrete_sequence=["#06b6d4"],
    )
    fig2.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Repos",
        yaxis_title=None,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── AI model usage ───────────────────────────────────────────────────────────
st.divider()
st.subheader("AI model usage across government repos")

ai_repos = [r for r in repos if r.get("domain") == "ai_ml" and r.get("ai_providers")]

if not ai_repos:
    st.caption("No AI provider data yet — run the pipeline to detect model usage.")
else:
    from collections import Counter

    tier_counts = Counter()
    provider_counts = Counter()

    for r in ai_repos:
        p = json.loads(r["ai_providers"])
        has_frontier    = bool(p.get("frontier"))
        has_open_weight = bool(p.get("open_weight"))
        has_framework   = bool(p.get("frameworks"))

        if has_frontier and has_open_weight:
            tier_counts["Both frontier & open weight"] += 1
        elif has_frontier:
            tier_counts["Frontier only"] += 1
        elif has_open_weight:
            tier_counts["Open weight only"] += 1
        elif has_framework:
            tier_counts["Framework only (no provider)"] += 1
        else:
            tier_counts["No provider detected"] += 1

        for name in p.get("frontier", []):
            provider_counts[name] += 1
        for name in p.get("open_weight", []):
            provider_counts[name] += 1
        for name in p.get("frameworks", []):
            provider_counts[name] += 1

    col_a, col_b = st.columns(2)

    with col_a:
        tier_df = pd.DataFrame(tier_counts.most_common(), columns=["tier", "repos"])
        tier_colours = {
            "Frontier only":                   "#f59e0b",
            "Open weight only":                "#10b981",
            "Both frontier & open weight":     "#6366f1",
            "Framework only (no provider)":    "#06b6d4",
            "No provider detected":            "#94a3b8",
        }
        fig_tier = px.pie(
            tier_df, values="repos", names="tier",
            hole=0.5,
            color="tier",
            color_discrete_map=tier_colours,
            title="Frontier vs open weight",
        )
        fig_tier.update_traces(textposition="outside", textinfo="label+percent")
        fig_tier.update_layout(showlegend=False, margin=dict(t=40, b=40, l=20, r=20))
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_b:
        prov_df = pd.DataFrame(provider_counts.most_common(12), columns=["provider", "repos"])
        label_map = {
            **{k: k.title() for k in FRONTIER},
            **{k: k.title() for k in OPEN_WEIGHT},
            **{k: k.title() for k in FRAMEWORKS},
            "aws-bedrock": "AWS Bedrock",
            "huggingface": "HuggingFace",
            "llamaindex": "LlamaIndex",
            "litellm": "LiteLLM",
            "langchain": "LangChain",
            "semantic-kernel": "Semantic Kernel",
        }
        prov_df["provider"] = prov_df["provider"].map(lambda x: label_map.get(x, x))

        def _colour(name: str) -> str:
            raw = name.lower().replace(" ", "-").replace("_", "-")
            if raw in {k.lower() for k in FRONTIER}:
                return "#f59e0b"
            if raw in {k.lower() for k in OPEN_WEIGHT}:
                return "#10b981"
            return "#06b6d4"

        prov_df["colour"] = prov_df["provider"].apply(_colour)
        fig_prov = px.bar(
            prov_df, x="repos", y="provider", orientation="h",
            color="colour", color_discrete_map="identity",
            title="Most-used AI providers / frameworks",
        )
        fig_prov.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
            xaxis_title="Repos",
            yaxis_title=None,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_prov, use_container_width=True)

    st.caption(
        "🟡 Frontier (OpenAI, Anthropic, Google, AWS Bedrock) · "
        "🟢 Open weight (Mistral, Cohere, HuggingFace, Ollama) · "
        "🔵 Framework (LangChain, LlamaIndex, LiteLLM…)"
    )

# ── Row 2: stars distribution + licence / CI health ──────────────────────────
st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("Stars distribution")
    stars_df = df[df["stars"] > 0]
    fig3 = px.histogram(
        stars_df, x="stars",
        nbins=50,
        log_y=True,
        color_discrete_sequence=["#8b5cf6"],
    )
    fig3.update_layout(
        xaxis_title="Stars",
        yaxis_title="Repos (log scale)",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Repo health by country")
    health = (
        df.groupby("country")
        .agg(
            licence_pct=("has_licence", "mean"),
            ci_pct=("has_ci", "mean"),
            repo_count=("id", "count"),
        )
        .reset_index()
        .sort_values("repo_count", ascending=False)
        .head(12)
    )
    health["licence_pct"] = (health["licence_pct"] * 100).round(1)
    health["ci_pct"] = (health["ci_pct"] * 100).round(1)

    fig4 = px.scatter(
        health,
        x="licence_pct",
        y="ci_pct",
        size="repo_count",
        text="country",
        color_discrete_sequence=["#10b981"],
        labels={"licence_pct": "Has licence (%)", "ci_pct": "Has CI (%)"},
    )
    fig4.update_traces(textposition="top center")
    fig4.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)
