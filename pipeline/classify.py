import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from mistralai.client.sdk import Mistral
from pydantic import BaseModel

from pipeline.store import DB_PATH, get_unclassified, update_classification

_PROMPT = (Path(__file__).parent / "prompts" / "classify.txt").read_text()
_MODEL = "mistral-small-latest"
_LOG_PATH = Path(__file__).parent.parent / "data" / "pipeline_log.jsonl"

# mistral-small-latest pricing: USD per 1M tokens
_PRICE_IN = 0.20
_PRICE_OUT = 0.60


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
    topics = ", ".join(repo.get("topics") or []) or "none"
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


def classify_repo(repo: dict, client: Mistral) -> ClassificationResult:
    result, _ = _call_model(repo, client)
    return result


def _log_batch(n_repos: int, input_tokens: int, output_tokens: int) -> None:
    cost = (input_tokens * _PRICE_IN + output_tokens * _PRICE_OUT) / 1_000_000
    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "repos_classified": n_repos,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def classify_batch(client: Mistral, limit: int = 50,
                   db_path: Path = DB_PATH) -> None:
    repos = get_unclassified(limit=limit, db_path=db_path)
    total_in = total_out = 0

    for repo in repos:
        result, usage = _call_model(repo, client)
        update_classification(repo["id"], result.model_dump(), db_path)
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens

    if repos:
        _log_batch(len(repos), total_in, total_out)
