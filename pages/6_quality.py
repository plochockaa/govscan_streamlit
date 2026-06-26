import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.store import get_eval_stats

st.set_page_config(page_title="Classification Quality — GovScan", layout="wide")

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
    return get_eval_stats()


st.title("Classification Quality")
st.caption(
    "LLM-as-judge: a Gemini evaluator independently reviews a sample of "
    "classified repositories each night and scores domain accuracy, summary "
    "quality, and confidence calibration."
)

stats = load()

if stats["total_evals"] == 0:
    st.info(
        "No evaluations yet — the pipeline hasn't run since the evaluation "
        "agent was added. Check back after the next nightly run."
    )
    st.stop()

# ── Headline metrics ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Repos Evaluated", f"{stats['total_evals']:,}")
c2.metric(
    "Avg Quality Score",
    f"{stats['avg_score']:.0%}" if stats["avg_score"] is not None else "—",
    help="Weighted: 60% domain correct, 30% summary quality, 10% confidence calibration",
)
c3.metric(
    "Domain Accuracy",
    f"{stats['pct_domain_correct']:.0%}" if stats["pct_domain_correct"] is not None else "—",
    help="% of evaluated repos where the evaluator agrees with the assigned domain",
)

st.divider()

# ── Score by domain ───────────────────────────────────────────────────────────
if stats["by_domain"]:
    df = pd.DataFrame(stats["by_domain"])
    df["domain_label"] = df["domain"].map(DOMAIN_LABELS).fillna(df["domain"])
    df["pct_correct_label"] = (df["pct_correct"] * 100).round(0).astype(int).astype(str) + "% correct"

    fig = px.bar(
        df.sort_values("avg_score"),
        x="avg_score",
        y="domain_label",
        orientation="h",
        color="avg_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        text="pct_correct_label",
        hover_data={"n": True, "avg_score": ":.2f", "pct_correct": ":.0%"},
        labels={"avg_score": "Avg Score", "domain_label": "", "n": "Repos Evaluated"},
        title="Quality Score by Domain",
        height=400,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Flagged repos ─────────────────────────────────────────────────────────────
flagged = stats["flagged"]
st.subheader(f"Flagged for Review ({len(flagged)} repos)")
st.caption("Repos where the evaluator disagrees with the assigned domain.")

if not flagged:
    st.success("No misclassifications detected in the evaluated sample.")
else:
    df_flag = pd.DataFrame(flagged)
    df_flag["domain_label"] = df_flag["domain"].map(DOMAIN_LABELS).fillna(df_flag["domain"])
    df_flag["suggested_label"] = df_flag["suggested_domain"].map(DOMAIN_LABELS).fillna(df_flag["suggested_domain"])
    df_flag["github_url"] = "https://github.com/" + df_flag["repo_id"]

    st.dataframe(
        df_flag[["repo_id", "country", "domain_label", "suggested_label", "overall_score", "reasoning", "github_url"]],
        column_config={
            "repo_id":         st.column_config.TextColumn("Repository"),
            "country":         st.column_config.TextColumn("Country"),
            "domain_label":    st.column_config.TextColumn("Assigned Domain"),
            "suggested_label": st.column_config.TextColumn("Suggested Domain"),
            "overall_score":   st.column_config.NumberColumn("Score", format="%.2f"),
            "reasoning":       st.column_config.TextColumn("Evaluator Reasoning", width="large"),
            "github_url":      st.column_config.LinkColumn("GitHub"),
        },
        use_container_width=True,
        hide_index=True,
    )
