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
# READ IN DATA FRAMES
# ----------------------

#df_all = pd.read_csv('all_gov_projects.csv')
df = pd.read_csv('filtered_repositories.csv')

# ----------------------
# LOAD STATIC CSV DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("govtech_data.csv")  # Ensure this file is in the same folder as app.py
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["year"] = df["updated_at"].dt.year
    return df

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="GovScan Prototype", layout="wide")
st.title("🌍 GovScan: Global GovTech Repo Aggregator")

# Load data from static CSV
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

