import json
import sqlite3

from google import genai
from google.genai import types

from pipeline.store import DB_PATH

_MODEL = "gemini-2.5-flash"

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

Policy area values: health, transport, benefits, tax, justice, education,
                    environment, cross_cutting, unknown.

Use query_repos to get accurate counts and facts before answering.
Always cite specific numbers. Keep answers concise but data-driven."""


def query_repos(sql: str) -> str:
    """Execute a SQL SELECT query against the GovScan repos table.
    Returns up to 50 rows as JSON. Only SELECT statements are permitted."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchmany(50)
        conn.close()
        return json.dumps([dict(r) for r in rows], default=str) if rows else "[]"
    except Exception as exc:
        return f"Query error: {exc}"


def get_cluster_members(cluster_id: int) -> str:
    """Return all repos in a similarity cluster (repos that do similar things
    across different governments). Returns id, name, country, domain, llm_summary,
    stars for each member, ordered by stars."""
    return query_repos(f"""
        SELECT id, name, country, domain, llm_summary, stars
        FROM repos WHERE cluster_id = {int(cluster_id)}
        ORDER BY stars DESC LIMIT 50
    """)


class QueryAgent:
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def ask(self, question: str, history: list[dict] | None = None) -> str:
        contents = []
        for msg in (history or []):
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=question)])
        )

        response = self._client.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                tools=[query_repos, get_cluster_members],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False,
                    maximum_remote_calls=5,
                ),
            ),
        )
        return response.text
