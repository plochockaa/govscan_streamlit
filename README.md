# GovScan — International Government GitHub Scanner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://govscan-interview.streamlit.app)

A live tool for scanning, filtering, and analysing open-source repositories published by government organisations worldwide. Built as a prototype to demonstrate what cross-government AI project intelligence could look like at scale.

---

## Demo

![GovScan demo](docs/demo.gif)

🔗 **[Live app → govscan-interview.streamlit.app](https://govscan-interview.streamlit.app)**

---

## What it does

Government organisations increasingly publish AI and data projects on GitHub, but there's no unified way to discover or compare what's being built across countries. GovScan pulls live repository data from government GitHub organisations across the UK, Canada, Singapore, the US, and the EU, and presents it in a filterable dashboard.

You can:
- Filter by programming language, organisation, country, and year
- Rank repositories by activity (stars, last update)
- Export filtered results as CSV for further analysis

---

## Architecture

    GitHub REST API
          │
          ▼
    load_data() [cached]     ← pulls repos + metadata per org
          │
          ▼
    pandas DataFrame         ← filtering, aggregation
          │
          ▼
    Streamlit UI             ← sidebar filters, table, CSV export

The app uses `@st.cache_data` to avoid re-fetching on every interaction. The GitHub token is stored in Streamlit Secrets — never hardcoded.

**Organisations currently tracked:**

| Organisation | Country |
|---|---|
| alphagov, i-dot-ai | UK |
| canada-ca | Canada |
| govtechsg | Singapore |
| GSA | USA |
| ec-europa | EU |
| opengovsg | Singapore |

---

## Running locally

```bash
git clone https://github.com/plochockaa/govscan_streamlit
cd govscan_streamlit
pip install -r requirements.txt
```

Add your GitHub token to `.streamlit/secrets.toml`:

```toml
GITHUB_TOKEN = "ghp_your_token_here"
```

Then:

```bash
streamlit run app.py
```

---

## What I'd build next

- **LLM classification** — tag each repo by AI maturity, domain, and deployment status from README content
- **Automated org discovery** — find government orgs via the GitHub API rather than a hardcoded list
- **Change tracking** — store snapshots over time to surface which governments are publishing more AI projects
- **Multilingual README parsing** — many EU and Asian government repos have non-English READMEs; an LLM layer could normalise these

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![GitHub API](https://img.shields.io/badge/GitHub_API-181717?style=flat&logo=github&logoColor=white)
