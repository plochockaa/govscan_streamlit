import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from pipeline.store import DB_PATH, get_unevaluated, store_eval

log = logging.getLogger(__name__)

_MODEL = "gemini-2.0-flash"
_LOG_PATH = Path(__file__).parent.parent / "data" / "pipeline_log.jsonl"

# Same pricing as classify.py (gemini-2.0-flash)
_PRICE_IN = 0.10
_PRICE_OUT = 0.40

_SYSTEM_PROMPT = """You are evaluating the quality of automated classifications of government GitHub repositories.
Given the raw repository data and the LLM classification, assess whether the classification is correct.

Valid domains:
  ai_ml               Machine learning, LLMs, AI assistants, NLP, neural networks, computer vision
  data_infrastructure ETL pipelines, data warehouses, dashboards, analytics platforms
  citizen_services    Public-facing government services: benefits, booking, forms, payments
  security            Authentication, identity, SSO, OAuth, vulnerability scanning, DMARC
  open_data           Datasets, open data portals, statistical releases, CKAN extensions
  devtools            CI/CD helpers, SDKs, design systems, linters, Terraform, boilerplates
  research            Academic or exploratory work, experiments, proofs-of-concept, simulations
  policy_tools        Legislative tracking, consultation platforms, procurement, regulatory compliance
  other               Does not fit any category above

Return ONLY valid JSON with exactly these fields — no prose, no markdown fences:

{
  "domain_correct":         true if the assigned domain is correct, false otherwise,
  "suggested_domain":       the domain you would assign if different, else null,
  "summary_quality":        integer 1–5 (1=wrong/missing, 3=adequate, 5=precise and informative),
  "confidence_appropriate": true if the confidence score matches the evidence quality,
  "reasoning":              one sentence explaining your assessment
}

Example output:
{"domain_correct":true,"suggested_domain":null,"summary_quality":4,"confidence_appropriate":true,"reasoning":"Correctly classified as citizen_services — the repo is a GOV.UK benefits application portal."}"""


class EvalResult(BaseModel):
    domain_correct: bool
    suggested_domain: Literal[
        "ai_ml", "data_infrastructure", "citizen_services", "security",
        "open_data", "devtools", "research", "policy_tools", "other",
    ] | None = None
    summary_quality: int          # 1–5
    confidence_appropriate: bool
    reasoning: str


def _compute_score(result: EvalResult) -> float:
    """Derive a single 0–1 score from the three eval dimensions."""
    return round(
        0.6 * float(result.domain_correct)
        + 0.3 * (result.summary_quality - 1) / 4   # normalise 1–5 → 0–1
        + 0.1 * float(result.confidence_appropriate),
        3,
    )


def _build_user_msg(repo: dict) -> str:
    raw_topics = repo.get("topics") or "[]"
    if isinstance(raw_topics, str):
        try:
            raw_topics = json.loads(raw_topics)
        except (json.JSONDecodeError, ValueError):
            raw_topics = []
    topics = ", ".join(raw_topics) or "none"
    readme = (repo.get("readme_text") or "")[:400]

    return (
        f"Repository:\n"
        f"name: {repo['name']}\n"
        f"description: {repo.get('description') or 'none'}\n"
        f"language: {repo.get('language') or 'unknown'}\n"
        f"topics: {topics}\n"
        f"readme: {readme}\n\n"
        f"Classification assigned:\n"
        f"domain: {repo['domain']}\n"
        f"maturity: {repo['maturity']}\n"
        f"policy_area: {repo['policy_area']}\n"
        f"summary: {repo['llm_summary']}\n"
        f"confidence: {repo['llm_confidence']}"
    )


def _call_model(repo: dict, client: genai.Client) -> tuple[EvalResult, object]:
    response = client.models.generate_content(
        model=_MODEL,
        contents=_build_user_msg(repo),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    result = EvalResult(**json.loads(response.text))
    return result, response.usage_metadata


def _log_eval_batch(n_repos: int, input_tokens: int, output_tokens: int) -> None:
    cost = (input_tokens * _PRICE_IN + output_tokens * _PRICE_OUT) / 1_000_000
    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "stage": "evaluate",
        "repos_evaluated": n_repos,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def evaluate_batch(
    client: genai.Client,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> dict:
    """LLM-as-judge: evaluate classification quality for a sample of repos."""
    repos = get_unevaluated(limit=limit, db_path=db_path)
    total_in = total_out = 0
    scores = []

    for repo in repos:
        try:
            result, usage = _call_model(repo, client)
            score = _compute_score(result)
            store_eval(
                repo_id=repo["id"],
                eval_model=_MODEL,
                domain_correct=result.domain_correct,
                suggested_domain=result.suggested_domain,
                summary_quality=result.summary_quality,
                confidence_ok=result.confidence_appropriate,
                overall_score=score,
                reasoning=result.reasoning,
                db_path=db_path,
            )
            scores.append(score)
            total_in += usage.prompt_token_count or 0
            total_out += usage.candidates_token_count or 0
            time.sleep(0.2)
        except Exception as exc:
            log.warning("Skipping eval for %s — %s", repo["id"], exc)
            time.sleep(1.0)

    if repos:
        _log_eval_batch(len(repos), total_in, total_out)

    cost = (total_in * _PRICE_IN + total_out * _PRICE_OUT) / 1_000_000
    return {
        "repos_evaluated": len(repos),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost, 6),
    }
