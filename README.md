# GovScan — International Government GitHub Scanner

A live intelligence tool that tracks, classifies, and compares open-source work published by government organisations on GitHub worldwide. The nightly pipeline fetches repos, classifies them using a Gemini-powered AI agent, embeds them for similarity search, and clusters them to surface duplicate efforts across countries. A natural-language query agent lets you interrogate the entire dataset through conversation.

🔗 **[Live app → govscan.streamlit.app](https://govscan.streamlit.app)**

![GovScan demo](https://raw.githubusercontent.com/plochockaa/govscan_streamlit/main/pages/govscan.gif)

---

## What it does

- **Classifies** every government repo by domain (AI/ML, citizen services, open data, security, etc.), policy area, and maturity using a Gemini agent — two-step reasoning: metadata first, README fetch if confidence is low
- **Answers natural-language questions** about government tech trends via a Gemini-powered query agent with SQL and similarity-search tools
- **Detects AI model usage** by scanning dependency files — distinguishes frontier models (OpenAI, Anthropic, AWS Bedrock) from open weight (Mistral, HuggingFace, Ollama)
- **Clusters similar repos** across governments to find where multiple countries built the same thing independently
- **Tracks 12,700+ repos** from 36 government organisations across 16 countries, updated nightly

---

## Pages

| Page | Description |
|---|---|
| **Overview** | Domain breakdown, repos by country |
| **Trends** | Repos over time, top languages, AI frontier vs open weight usage |
| **Similarity** | Expandable cluster cards — countries, repo links, LLM summaries, domain filters |
| **Search** | Full-text search across name / description / LLM summary with sidebar filters |
| **Ask** | Natural-language chat agent — ask questions, get data-backed answers |

---

## Architecture

```
GitHub REST API
      │
      ▼
pipeline/fetch.py        ← paginated org scraping, rate-limit handling
      │
      ▼
pipeline/store.py        ← SQLite (repos, embeddings, clusters, ai_providers)
      │
      ├── pipeline/classify.py   ← Gemini agent → domain / maturity / policy_area
      │                             (2-step: metadata first; fetches README if confidence < 0.65)
      │
      ├── pipeline/embed.py      ← fastembed (BAAI/bge-small-en-v1.5, local ONNX)
      │
      ├── pipeline/cluster.py    ← KMeans on embeddings → cluster_id
      │
      └── pipeline/detect.py     ← scan requirements.txt / package.json for AI SDKs
                                    tier as frontier / open weight / frameworks

data/govscan.db          ← committed to repo nightly by CI
      │
      ▼
Streamlit multipage app  ← reads DB; Ask page also calls Gemini query agent at runtime
      │
      └── pipeline/query_agent.py ← Gemini 2.5 Flash + SQL tools → natural-language answers
```

The data pipeline is fully offline — all heavy processing runs in CI overnight. The Ask page is the only part that calls an LLM at runtime.

---

## Organisations tracked

| Country | Organisations |
|---|---|
| UK | alphagov, i-dot-ai, co-cddo, nhsengland, DWPDigital, hmrc, ministryofjustice, ScottishGovernment |
| USA | GSA, 18F, uswds, CDCgov, USDS |
| France | betagouv, numerique-gouv, etalab |
| Canada | canada-ca, cds-snc |
| Singapore | govtechsg, opengovsg |
| Germany | digitalservicebund |
| Netherlands | minbzk, nl-design-system |
| Australia | AusDTO, govau, ServiceNSW |
| New Zealand | ServiceInnovationLab, GOVTNZ |
| Sweden | diggsweden |
| Denmark | digst |
| Estonia | e-gov |
| India | egovernments, mosip |
| Brazil | servicosgovbr |
| Taiwan | g0v |
| EU | ec-europa |

---

## Running locally

```bash
git clone https://github.com/plochockaa/govscan_streamlit
cd govscan_streamlit
uv sync                        # installs app deps only (streamlit, pandas, plotly)
uv sync --extra pipeline       # also installs pipeline deps (google-genai, fastembed, etc.)
```

Copy `.env.example` and fill in your tokens:

```bash
cp .env.example .env
# add GH_TOKEN and GEMINI_API_KEY
```

Run the app:

```bash
uv run streamlit run app.py
```

Run the full pipeline manually:

```bash
uv run python -m pipeline.run
```

---

## CI / CD

- **`test.yml`** — runs pytest on every push to main
- **`pipeline.yml`** — nightly at 01:00 CEST: fetch → classify → embed → cluster → detect AI providers, then commits the updated `data/govscan.db` back to the repo, which triggers a Streamlit Cloud redeploy automatically

Secrets required in GitHub repo settings: `GEMINI_API_KEY`, `GH_TOKEN` (for reading public repos past the unauthenticated rate limit).

---

## Tech stack

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)
