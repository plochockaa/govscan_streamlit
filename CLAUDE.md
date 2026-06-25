# CLAUDE.md

## Project: GovScan

GovScan tracks open-source work across government GitHub organisations worldwide. It detects trends, classifies repositories by domain, spots duplicate efforts across countries, and surfaces AI/ML adoption in the public sector.

Live app: https://govscan.streamlit.app  
Source: https://github.com/plochockaa/govscan_streamlit

---

## Architecture

```
pipeline/           Nightly data pipeline (run locally / CI)
  fetch.py          GitHub API → raw repo data + READMEs
  classify.py       Mistral LLM → domain / maturity / policy_area / summary
  embed.py          fastembed (BAAI/bge-small-en-v1.5) → float32 blobs
  cluster.py        KMeans on normalised embeddings → cluster_id
  detect.py         Dep-file + text scan → AI provider tiers
  store.py          All SQLite reads/writes — single source of truth for DB ops
  run.py            Orchestrates all stages in order
  rescan.py         One-off rescanning utilities
  prompts/
    classify.txt    System prompt for Mistral classification

app.py              Streamlit landing page (metrics + nav)
pages/
  1_overview.py     Domain breakdown, repos by country
  2_trends.py       Activity over time, language breakdown
  3_similarity.py   Clusters of similar repos across governments
  4_search.py       Browse / filter all classified repos
  5_ask.py          Gemini-powered chat agent — query the DB in natural language

config.py           ORGS dict: country → list of GitHub org names
data/
  govscan.db        SQLite database (committed nightly by CI; Streamlit Cloud redeploys on each commit)
  pipeline_log.jsonl  Per-run cost/token accounting
```

**Data flow:** `fetch → readme fetch → classify → embed → cluster → detect`  
Each stage is idempotent; `run.py` skips rows already processed.

---

## Tech stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data / charts | pandas, plotly |
| Classification LLM | Gemini (`gemini-2.0-flash`) via `google-genai` SDK |
| Query Agent LLM | Gemini (`gemini-2.5-flash`) — powers the Ask page chat agent |
| Embeddings | `fastembed` — `BAAI/bge-small-en-v1.5` (local, no API cost) |
| Clustering | `scikit-learn` KMeans |
| Database | SQLite (WAL mode) — file at `data/govscan.db` |
| Package manager | `uv` |

Streamlit dependencies: `streamlit`, `pandas`, `plotly` (in `requirements.txt`).  
Pipeline dependencies: extras group `[pipeline]` in `pyproject.toml`.

---

## Database schema (`repos` table)

| Column | Notes |
|---|---|
| `id` | `org/repo-name` — primary key |
| `org`, `country` | From `config.ORGS` |
| `domain` | Null until classified; one of the 9 domain values in `classify.txt` |
| `maturity` | `prototype / active / maintained / archived` |
| `policy_area` | `health / transport / benefits / tax / justice / education / environment / cross_cutting / unknown` |
| `llm_summary` | One-sentence description from the LLM |
| `llm_confidence` | Float 0–1; `update_classification` sets it; `detect.py` may force it to 1.0 on reclassification |
| `embedding` | Raw `float32` bytes (384-dim); null until embedded |
| `cluster_id` | Integer ≥ 0 after clustering; heuristic `n_clusters = max(2, min(50, n//8))` |
| `ai_providers` | JSON: `{"frontier": [...], "open_weight": [...], "frameworks": [...]}` |
| `readme_text` | First 3000 chars; fetched before classification, never overwritten if already set |

The `pipeline_log` table in the DB and `data/pipeline_log.jsonl` both track per-run token usage and cost.

---

## Environment variables

```
GH_TOKEN          GitHub personal access token (repo:read scope)
GEMINI_API_KEY    Gemini API key from aistudio.google.com
```

Store in a `.env` file at the repo root for local runs — `pipeline/run.py` calls `load_dotenv()`.  
The Streamlit app reads only from the DB; it needs no API keys.

---

## Running locally

```bash
# Install all dependencies (app + pipeline)
uv sync --extra pipeline

# Run the pipeline once
python -m pipeline.run

# Start the Streamlit app
streamlit run app.py

# Run tests
pytest
```

---

## Adding a new government organisation

Edit `config.py` — add the GitHub org name to the appropriate country list.  
The next pipeline run will fetch and classify all repos for that org automatically.

---

## Classification domains

`ai_ml` | `data_infrastructure` | `citizen_services` | `security` | `open_data` | `devtools` | `research` | `policy_tools` | `other`

Full definitions live in `pipeline/prompts/classify.txt`. If classification results look wrong for a new org type, edit that prompt — do not touch the `Domain` enum in `classify.py` without updating both.

---

## AI provider detection (`detect.py`)

Two-pass approach:
1. **Dep-file scan** — fetches `requirements.txt`, `pyproject.toml`, `package.json`, `go.mod`, `Gemfile`, `composer.json`, etc. via GitHub API. Looks for known package names across three tiers: `frontier` (OpenAI, Anthropic, Google, AWS Bedrock), `open_weight` (Mistral, HuggingFace, Ollama, vLLM, …), `frameworks` (LangChain, LlamaIndex, DSPy, …).
2. **Text scan** (`detect_from_text`) — regex patterns against README text already in the DB; zero extra API calls.

After main detection, `run.py` re-scans non-`ai_ml` repos for any AI SDK usage and reclassifies them if found.

---

## Coding guidelines

**1. Think before coding — don't assume, surface tradeoffs**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

**2. Simplicity first — minimum code that solves the problem**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

**3. Surgical changes — touch only what you must**

- Don't improve adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove only imports/variables YOUR changes made unused.

**4. Goal-driven execution — define success criteria, loop until verified**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

---

**These guidelines work when:** diffs are clean, changes trace to the request, and clarifying questions come before implementation rather than after mistakes.
