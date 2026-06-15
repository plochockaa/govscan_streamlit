import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.store import get_all_repos

st.set_page_config(page_title="Overview — GovScan", layout="wide")

DOMAIN_LABELS = {
    "ai_ml":              "AI / ML",
    "data_infrastructure": "Data Infrastructure",
    "citizen_services":   "Citizen Services",
    "security":           "Security",
    "open_data":          "Open Data",
    "devtools":           "Developer Tools",
    "research":           "Research",
    "policy_tools":       "Policy Tools",
    "other":              "Other",
}

DOMAIN_COLOURS = {
    "AI / ML":             "#6366f1",
    "Data Infrastructure": "#06b6d4",
    "Citizen Services":    "#10b981",
    "Security":            "#f59e0b",
    "Open Data":           "#3b82f6",
    "Developer Tools":     "#8b5cf6",
    "Research":            "#ec4899",
    "Policy Tools":        "#14b8a6",
    "Other":               "#94a3b8",
}


@st.cache_data(ttl=3600)
def load():
    repos = get_all_repos()
    return [r for r in repos if r["domain"]]


st.title("📊 Overview")

repos = load()

if not repos:
    st.info("No classified repos yet — the pipeline is still running.")
    st.stop()

df = pd.DataFrame(repos)
df["domain_label"] = df["domain"].map(DOMAIN_LABELS).fillna("Other")

# ── Row 1: donut + bar ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Domain breakdown")
    counts = df["domain_label"].value_counts().reset_index()
    counts.columns = ["domain", "count"]
    fig = px.pie(
        counts,
        values="count",
        names="domain",
        hole=0.55,
        color="domain",
        color_discrete_map=DOMAIN_COLOURS,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(showlegend=False, margin=dict(t=20, b=40, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI / ML repos by country")
    ai_df = df[df["domain"] == "ai_ml"]
    if ai_df.empty:
        # fall back to all classified repos by country when AI/ML count is tiny
        st.caption("Showing all classified repos by country (AI / ML count too small to chart alone)")
        country_counts = df["country"].value_counts().reset_index()
        country_counts.columns = ["country", "count"]
    else:
        country_counts = ai_df["country"].value_counts().reset_index()
        country_counts.columns = ["country", "count"]

    fig2 = px.bar(
        country_counts,
        x="count",
        y="country",
        orientation="h",
        color_discrete_sequence=["#6366f1"],
    )
    fig2.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Repos",
        yaxis_title=None,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: stacked bar country × domain ──────────────────────────────────────
st.divider()
st.subheader("Domain mix by country")

stacked = (
    df.groupby(["country", "domain_label"])
    .size()
    .reset_index(name="count")
)
country_order = (
    df["country"].value_counts().index.tolist()
)
fig3 = px.bar(
    stacked,
    x="count",
    y="country",
    color="domain_label",
    orientation="h",
    color_discrete_map=DOMAIN_COLOURS,
    category_orders={"country": list(reversed(country_order))},
    labels={"domain_label": "Domain", "count": "Repos", "country": ""},
)
fig3.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, b=20, l=20, r=20),
    xaxis_title="Classified repos",
)
st.plotly_chart(fig3, use_container_width=True)
