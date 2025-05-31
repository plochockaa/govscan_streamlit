import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ----------------------
# CONFIGURATION
# ----------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # Store securely in Streamlit secrets
GITHUB_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
GOV_ORGS = [
    "datagovsg", "govau", "canada-ca", "UKHomeOffice", "italia", "govbr", "gov-si"
]

# ----------------------
# FUNCTIONS
# ----------------------
def fetch_repos(org):
    url = f"https://api.github.com/orgs/{org}/repos?per_page=100"
    response = requests.get(url, headers=GITHUB_HEADERS)
    if response.status_code != 200:
        return []
    return response.json()

def process_repos(repos):
    data = []
    for repo in repos:
        data.append({
            "name": repo["name"],
            "description": repo.get("description", ""),
            "language": repo.get("language"),
            "stars": repo.get("stargazers_count"),
            "updated_at": repo.get("updated_at"),
            "html_url": repo.get("html_url"),
            "org": repo["owner"]["login"]
        })
    return pd.DataFrame(data)

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="GovScan Prototype", layout="wide")
st.title("🌍 GovScan: Global GovTech Repo Aggregator")

if "repo_df" not in st.session_state:
    all_repos = []
    for org in GOV_ORGS:
        with st.spinner(f"Fetching repos from {org}..."):
            org_repos = fetch_repos(org)
            all_repos.extend(org_repos)
    st.session_state.repo_df = process_repos(all_repos)

# Display filters
st.sidebar.header("Filters")
language_filter = st.sidebar.multiselect("Language", options=st.session_state.repo_df["language"].dropna().unique())
org_filter = st.sidebar.multiselect("Organization", options=GOV_ORGS)

# Apply filters
df = st.session_state.repo_df.copy()
if language_filter:
    df = df[df["language"].isin(language_filter)]
if org_filter:
    df = df[df["org"].isin(org_filter)]

# Display table
st.subheader("📊 Repository Overview")
st.dataframe(df.sort_values("stars", ascending=False).reset_index(drop=True), use_container_width=True)

# Show insights
st.subheader("🔍 Quick Insights")
st.markdown(f"**Total Repos Scanned:** {len(st.session_state.repo_df)}")
st.markdown(f"**Most Common Language:** {st.session_state.repo_df['language'].mode().values[0]}")
latest_update = pd.to_datetime(st.session_state.repo_df['updated_at']).max()
st.markdown(f"**Last Repo Update Detected:** {latest_update.strftime('%Y-%m-%d')}")

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
CATEGORIES = ["Health", "Justice", "Education", "Infrastructure", "AI/Automation", "Cybersecurity"]

def classify_description(desc):
    result = classifier(desc, CATEGORIES)
    return result["labels"][0]  # Most likely category

df['category'] = df['description'].fillna("").apply(classify_description)

st.download_button("Download CSV", df.to_csv(index=False), file_name="govscan_repos.csv")

