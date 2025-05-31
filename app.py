import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ----------------------
# CONFIGURATION
# ----------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # Store securely in Streamlit secrets

@st.cache_data
def load_data():
    orgs = ["alphagov", "i-dot-ai", "canada-ca", "govtechsg", "GSA", "ec-europa", "opengovsg"]  # Example gov GitHub orgs
    all_repos = []

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    for org in orgs:
        url = f"https://api.github.com/orgs/{org}/repos?per_page=100"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            repos = response.json()
            for repo in repos:
                readme_url = f"https://api.github.com/repos/{org}/{repo['name']}/readme"
                readme_resp = requests.get(readme_url, headers=headers)
                readme = readme_resp.json().get("content", "") if readme_resp.status_code == 200 else ""

                country = "UK" if org in ["alphagov", "i-dot-ai"] else "not-UK"

                all_repos.append({
                    "language": repo["language"],
                    "org": org,
                    "updated_at": repo["updated_at"],
                    "year": pd.to_datetime(repo["updated_at"]).year,
                    "stars": repo["stargazers_count"],
                    "country": country,
                    "description": repo["description"],
                    "readme": readme,
                })
        else:
            st.warning(f"Failed to fetch repos for {org}: {response.status_code}")

    return pd.DataFrame(all_repos)

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="Gov Scan Prototype", layout="wide")
st.title("🌍 Government Github Scanner: classify international github projects")

# Load dynamic data from GitHub API
df = load_data()

# ----------------------
# FILTERS
# ----------------------
st.sidebar.header("Filters")
language_filter = st.sidebar.multiselect("Language", options=sorted(df["language"].dropna().unique()))
org_filter = st.sidebar.multiselect("Organization", options=sorted(df["org"].dropna().unique()))
years = sorted(df["year"].dropna().unique(), reverse=True)
selected_years = st.sidebar.multiselect("Year Updated", options=years, default=years)

# Apply filters
if language_filter:
    df = df[df["language"].isin(language_filter)]
if org_filter:
    df = df[df["org"].isin(org_filter)]
if selected_years:
    df = df[df["year"].isin(selected_years)]

# ----------------------
# DISPLAY TABLE
# ----------------------
st.subheader("📊 Repository Overview")
st.dataframe(df.sort_values("stars", ascending=False).reset_index(drop=True), use_container_width=True)

# ----------------------
# INSIGHTS
# ----------------------
st.subheader("🔍 Quick Insights")
st.markdown(f"**Total Repos Scanned:** {len(df)}")
if not df.empty:
    st.markdown(f"**Most Common Language:** {df['language'].mode().values[0]}")
    latest_update = pd.to_datetime(df['updated_at']).max()
    st.markdown(f"**Last Repo Update Detected:** {latest_update.strftime('%Y-%m-%d')}")
else:
    st.markdown("No data to display based on current filters.")

# ----------------------
# EXPORT DOWNLOAD
# ----------------------
st.subheader("⬇️ Download Filtered Data")
export_cols = ["language", "org", "year", "stars", "country", "description", "readme"]
export_df = df[export_cols].copy()
st.download_button(
    label="Download CSV",
    data=export_df.to_csv(index=False),
    file_name="govtech_filtered_export.csv",
    mime="text/csv"
) 
