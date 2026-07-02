import json
import sqlite3

import requests

from pipeline.store import DB_PATH

_MODEL = "gemini-2.5-flash"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_SYSTEM_PROMPT = """You are GovScan Intelligence, an AI analyst specialising in government open-source technology.
You have access to a database of repositories from government organisations across 16 countries.

The repos table has these columns:
  id (TEXT, format 'org/repo-name'), org, country, name, description, language,
  stars (INTEGER), forks (INTEGER), domain, maturity, policy_area,
  llm_summary, llm_confidence (REAL 0–1), cluster_id (INTEGER),
  ai_providers (JSON TEXT with keys frontier/open_weight/frameworks),
  topics (JSON TEXT array), created_at, updated_at.

Domain values: ai_ml, data_infrastructure, citizen_services, security, open_data,
               devtools, research, policy_tools, other.

Always cite specific numbers and country names. Keep answers concise but data-driven."""

_SQL_PROMPT = """You are a SQL expert. Given a question about government open-source repositories,
write a single SQL SELECT query against the repos table.

The repos table columns: id, org, country, name, description, language, stars (INTEGER),
forks (INTEGER), domain, maturity, policy_area, llm_summary, llm_confidence (REAL 0-1),
cluster_id (INTEGER), ai_providers (JSON TEXT), topics (JSON TEXT), created_at, updated_at.

Domain values: ai_ml, data_infrastructure, citizen_services, security, open_data,
               devtools, research, policy_tools, other.
Policy area values: health, transport, benefits, tax, justice, education, environment,
                    cross_cutting, unknown.

IMPORTANT: When selecting individual repositories (not just counts/aggregations), always include:
  - id          (used to build the GitHub URL: https://github.com/{{id}})
  - name
  - country
  - llm_summary (one-sentence description)
  - ai_providers (JSON column — include it whenever AI models are relevant)

For questions about AI model usage, filter with:
  json_array_length(json_extract(ai_providers, '$.open_weight')) > 0   (open-weight models)
  json_array_length(json_extract(ai_providers, '$.frontier'))    > 0   (frontier models)
  json_array_length(json_extract(ai_providers, '$.frameworks'))  > 0   (AI frameworks)

Return ONLY the SQL query. No markdown fences, no explanation.

Question: {question}"""

_SYNTHESIS_PROMPT = """{system}

{history}Question: {question}
SQL used: {sql}
Results (JSON): {results}

Format your answer as markdown:
- Start with a one-sentence summary of the finding.
- Group results by country using bold headers (e.g. **United Kingdom**).
- For each repository, write a bullet: [repo-name](https://github.com/{{id}}) — {{what it does}}
  - If ai_providers is present, add: Models detected: {{open_weight list}}
- End with a brief insight or pattern you notice across countries.
- Only cite repositories from the results above — do not invent any."""


def _call_gemini(api_key: str, prompt: str) -> str:
    url = f"{_API_BASE}/models/{_MODEL}:generateContent?key={api_key}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _extract_sql(raw: str) -> str:
    """Strip markdown fences if the model added them."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
    return raw.strip()


def _safe_query(sql: str) -> str:
    if not sql.strip().upper().startswith("SELECT"):
        return "[]"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchmany(50)
        conn.close()
        return json.dumps([dict(r) for r in rows], default=str) if rows else "[]"
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _enrich_results(results_json: str) -> str:
    """Parse results and make ai_providers human-readable; add github_url."""
    try:
        rows = json.loads(results_json)
    except (json.JSONDecodeError, ValueError):
        return results_json

    if not isinstance(rows, list):
        return results_json

    for row in rows:
        if "id" in row:
            row["github_url"] = f"https://github.com/{row['id']}"
        if "ai_providers" in row and row["ai_providers"]:
            try:
                providers = json.loads(row["ai_providers"])
                row["ai_providers"] = providers
            except (json.JSONDecodeError, ValueError):
                pass

    return json.dumps(rows, indent=2)


class QueryAgent:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def ask(self, question: str, history: list[dict] | None = None) -> str:
        # Step 1 — generate SQL from the question
        sql_raw = _call_gemini(self._api_key, _SQL_PROMPT.format(question=question))
        sql = _extract_sql(sql_raw)

        # Step 2 — execute against the DB
        results_raw = _safe_query(sql)

        # Step 3 — enrich results (GitHub URLs, parsed ai_providers JSON)
        results = _enrich_results(results_raw)

        # Step 4 — synthesise with evidence formatting
        history_block = ""
        if history:
            recent = history[-4:]
            history_block = (
                "Recent conversation:\n"
                + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
                + "\n\n"
            )

        synthesis = _SYNTHESIS_PROMPT.format(
            system=_SYSTEM_PROMPT,
            history=history_block,
            question=question,
            sql=sql,
            results=results,
        )
        return _call_gemini(self._api_key, synthesis)
