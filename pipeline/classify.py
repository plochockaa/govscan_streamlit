import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from mistralai.client import Mistral
from pydantic import BaseModel

from pipeline.fetch import fetch_readme
from pipeline.store import DB_PATH, get_unclassified, update_classification

log = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent / "prompts" / "classify.txt").read_text()
_MODEL = "open-mistral-nemo"
_LOG_PATH = Path(__file__).parent.parent / "data" / "pipeline_log.jsonl"

# open-mistral-nemo pricing: USD per 1M tokens (free tier = $0, paid = $0.15)
_PRICE_IN = 0.15
_PRICE_OUT = 0.15

# Confidence threshold below which the agent fetches the README before reclassifying
_CONFIDENCE_THRESHOLD = 0.65


class Domain(str, Enum):
    AI_ML = "ai_ml"
    DATA_INFRA = "data_infrastructure"
    CITIZEN_SERVICES = "citizen_services"
    SECURITY = "security"
    OPEN_DATA = "open_data"
    DEVTOOLS = "devtools"
    RESEARCH = "research"
    POLICY_TOOLS = "policy_tools"
    OTHER = "other"


class ClassificationResult(BaseModel):
    domain: Domain
    maturity: Literal["prototype", "active", "maintained", "archived"]
    policy_area: Literal["health", "transport", "benefits", "tax",
                         "justice", "education", "environment",
                         "cross_cutting", "unknown"]
    summary: str
    confidence: float


def _build_user_msg(repo: dict) -> str:
    raw_topics = repo.get("topics") or "[]"
    if isinstance(raw_topics, str):
        try:
            raw_topics = json.loads(raw_topics)
        except (json.JSONDecodeError, ValueError):
            raw_topics = []
    topics = ", ".join(raw_topics) or "none"
    readme = (repo.get("readme_text") or "")[:800]
    return (
        f"name: {repo['name']}\n"
        f"description: {repo.get('description') or 'none'}\n"
        f"language: {repo.get('language') or 'unknown'}\n"
        f"topics: {topics}\n"
        f"readme: {readme}"
    )


def _call_model(repo: dict, client: Mistral) -> tuple[ClassificationResult, object]:
    response = client.chat.complete(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": _build_user_msg(repo)},
        ],
        response_format={"type": "json_object"},
    )
    result = ClassificationResult(**json.loads(response.choices[0].message.content))
    return result, response.usage


def _classify_agent(
    repo: dict,
    client: Mistral,
    gh_headers: dict | None,
) -> tuple[ClassificationResult, int, int, int]:
    """
    Two-step classification agent.

    Step 1 — classify from whatever metadata is available (name, description,
              language, topics, and readme_text if already in the DB).
    Step 2 — if confidence is below the threshold AND the repo has no readme yet,
              fetch the README from GitHub (one tool call) and reclassify.

    Returns (result, prompt_tokens, output_tokens, readme_fetches).
    """
    result, usage = _call_model(repo, client)
    prompt_tokens = usage.prompt_tokens or 0
    output_tokens = usage.completion_tokens or 0

    needs_more_context = (
        result.confidence < _CONFIDENCE_THRESHOLD
        and not repo.get("readme_text")
        and gh_headers is not None
    )
    if not needs_more_context:
        return result, prompt_tokens, output_tokens, 0

    # Tool call: fetch README to improve confidence
    org, name = repo["id"].split("/", 1)
    readme_text = fetch_readme(org, name, gh_headers)
    if not readme_text:
        return result, prompt_tokens, output_tokens, 1

    enriched = {**repo, "readme_text": readme_text}
    result, usage2 = _call_model(enriched, client)
    prompt_tokens += usage2.prompt_tokens or 0
    output_tokens += usage2.completion_tokens or 0
    return result, prompt_tokens, output_tokens, 1


def classify_repo(repo: dict, client: Mistral) -> ClassificationResult:
    """Classify a single repo without the agent loop (no GitHub calls)."""
    result, _ = _call_model(repo, client)
    return result


def _log_batch(
    n_repos: int, input_tokens: int, output_tokens: int, tool_calls: int
) -> None:
    cost = (input_tokens * _PRICE_IN + output_tokens * _PRICE_OUT) / 1_000_000
    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "repos_classified": n_repos,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "readme_fetches": tool_calls,
    }
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def classify_batch(
    client: Mistral,
    limit: int = 500,
    db_path: Path = DB_PATH,
    gh_headers: dict | None = None,
) -> dict:
    repos = get_unclassified(limit=limit, db_path=db_path)
    total_in = total_out = total_tool_calls = 0
    first_error: str | None = None

    for repo in repos:
        try:
            result, in_tok, out_tok, tool_calls = _classify_agent(
                repo, client, gh_headers
            )
            update_classification(repo["id"], result.model_dump(), db_path)
            total_in += in_tok
            total_out += out_tok
            total_tool_calls += tool_calls
            time.sleep(0.2)
        except Exception as exc:
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            log.warning("Skipping %s — classification failed: %s", repo["id"], exc)
            time.sleep(1.0)

    if repos:
        _log_batch(len(repos), total_in, total_out, total_tool_calls)

    cost = (total_in * _PRICE_IN + total_out * _PRICE_OUT) / 1_000_000
    return {
        "repos_classified": len(repos),
        "input_tokens":     total_in,
        "output_tokens":    total_out,
        "cost_usd":         round(cost, 6),
        "readme_fetches":   total_tool_calls,
        "error":            first_error,
    }
