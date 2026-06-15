import streamlit as st

from pipeline.store import get_all_repos, get_duplicate_efforts, get_repos_by_cluster

st.set_page_config(page_title="Similarity — GovScan", layout="wide")

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

POLICY_LABELS = {
    "health": "Health", "transport": "Transport", "benefits": "Benefits",
    "tax": "Tax", "justice": "Justice", "education": "Education",
    "environment": "Environment", "cross_cutting": "Cross-cutting",
    "unknown": "Unknown",
}


@st.cache_data(ttl=3600)
def load_clusters():
    return get_duplicate_efforts()


@st.cache_data(ttl=3600)
def load_cluster_repos(cluster_id: int):
    return get_repos_by_cluster(cluster_id)


st.title("🔁 Similar efforts across governments")
st.markdown(
    "Clusters where **multiple countries built the same thing**. "
    "Each card is a group of repos that are semantically similar."
)

clusters = load_clusters()

if not clusters:
    st.info(
        "No clusters found yet. The similarity clustering step hasn't run — "
        "once repos are embedded, run the clustering pipeline to populate this page.",
        icon="ℹ️",
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
all_repos = get_all_repos()
classified = {r["id"]: r for r in all_repos if r["domain"]}

# Gather domains / policy areas that appear in cluster members
cluster_meta: dict[int, dict] = {}
for c in clusters:
    repos = load_cluster_repos(c["cluster_id"])
    domains = sorted({classified[r["id"]]["domain"] for r in repos if r["id"] in classified})
    policies = sorted({classified[r["id"]]["policy_area"] for r in repos if r["id"] in classified and classified[r["id"]]["policy_area"]})
    cluster_meta[c["cluster_id"]] = {"repos": repos, "domains": domains, "policies": policies}

all_domains = sorted({d for m in cluster_meta.values() for d in m["domains"]})
all_policies = sorted({p for m in cluster_meta.values() for p in m["policies"]})

with st.sidebar:
    st.header("Filter clusters")
    sel_domains = st.multiselect(
        "Domain",
        options=all_domains,
        format_func=lambda x: DOMAIN_LABELS.get(x, x),
    )
    sel_policies = st.multiselect(
        "Policy area",
        options=all_policies,
        format_func=lambda x: POLICY_LABELS.get(x, x),
    )
    min_countries = st.slider("Min countries in cluster", min_value=2, max_value=10, value=2)

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = []
for c in clusters:
    meta = cluster_meta[c["cluster_id"]]
    if c["country_count"] < min_countries:
        continue
    if sel_domains and not any(d in meta["domains"] for d in sel_domains):
        continue
    if sel_policies and not any(p in meta["policies"] for p in sel_policies):
        continue
    filtered.append(c)

st.caption(f"Showing {len(filtered)} of {len(clusters)} clusters")

if not filtered:
    st.warning("No clusters match the current filters.")
    st.stop()

# ── Cluster cards ─────────────────────────────────────────────────────────────
for c in filtered:
    meta = cluster_meta[c["cluster_id"]]
    repos = meta["repos"]
    countries = c["countries"].split(",")
    country_flags = " · ".join(sorted(countries))
    domain_pills = " · ".join(DOMAIN_LABELS.get(d, d) for d in meta["domains"])

    header = f"**{country_flags}** — {c['repo_count']} repos across {c['country_count']} countries"
    if domain_pills:
        header += f" · _{domain_pills}_"

    with st.expander(header):
        for repo in repos:
            info = classified.get(repo["id"], {})
            org, name = repo["id"].split("/", 1)
            gh_url = f"https://github.com/{repo['id']}"
            summary = info.get("llm_summary") or repo.get("description") or ""
            policy = POLICY_LABELS.get(info.get("policy_area"), "")
            badges = " · ".join(filter(None, [
                DOMAIN_LABELS.get(info.get("domain"), ""),
                policy,
                repo.get("country", ""),
            ]))
            st.markdown(
                f"**[{name}]({gh_url})** `{org}`  \n"
                f"{summary}  \n"
                f"<small>{badges} · ⭐ {repo.get('stars', 0):,}</small>",
                unsafe_allow_html=True,
            )
            st.divider()
