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

Return ONLY the SQL query. No markdown fences, no explanation.

Question: {question}"""


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


class QueryAgent:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def ask(self, question: str, history: list[dict] | None = None) -> str:
        # Step 1 — generate SQL from the question
        sql_raw = _call_gemini(self._api_key, _SQL_PROMPT.format(question=question))
        sql = _extract_sql(sql_raw)

        # Step 2 — execute against the DB
        results = _safe_query(sql)

        # Step 3 — synthesise a natural-language answer
        history_block = ""
        if history:
            recent = history[-4:]
            history_block = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in recent
            )
            history_block = f"Recent conversation:\n{history_block}\n\n"

        synthesis = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{history_block}"
            f"Question: {question}\n"
            f"SQL used: {sql}\n"
            f"Results (JSON): {results}\n\n"
            "Answer clearly and concisely using the data above. "
            "Cite specific numbers and countries."
        )
        return _call_gemini(self._api_key, synthesis)
