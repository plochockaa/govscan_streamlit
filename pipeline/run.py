import logging
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

from config import ORGS
from pipeline.classify import classify_batch
from pipeline.cluster import cluster_repos
from pipeline.detect import detect_ai_providers, detect_from_text
from pipeline.embed import embed_and_store
from pipeline.fetch import fetch_org_repos, fetch_readme
from pipeline.store import (get_ai_ml_empty_providers, get_ai_ml_repos,
                             get_connection, get_missing_readme,
                             get_undetected_classified, init_db,
                             log_pipeline_run, update_ai_providers,
                             update_readme, upsert_repo)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _normalize(raw: dict, org: str, country: str) -> dict:
    return {
        "id":          raw["full_name"],
        "org":         org,
        "country":     country,
        "name":        raw["name"],
        "description": raw.get("description"),
        "readme_text": None,
        "language":    raw.get("language"),
        "stars":       raw.get("stargazers_count", 0),
        "forks":       raw.get("forks_count", 0),
        "open_issues": raw.get("open_issues_count", 0),
        "has_licence": int(raw.get("license") is not None),
        "has_ci":      0,
        "topics":      raw.get("topics", []),
        "created_at":  raw.get("created_at"),
        "updated_at":  raw.get("updated_at"),
    }


def run() -> None:
    gh_token = os.environ["GH_TOKEN"]
    gemini_key = os.environ["GEMINI_API_KEY"]

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github+json",
    }
    client = genai.Client(api_key=gemini_key)

    init_db()

    # --- Fetch ---
    total_fetched = 0
    for country, orgs in ORGS.items():
        for org in orgs:
            log.info("Fetching %s (%s)", org, country)
            try:
                repos = fetch_org_repos(org, headers)
                for raw in repos:
                    upsert_repo(_normalize(raw, org, country))
                total_fetched += len(repos)
                log.info("  %d repos upserted", len(repos))
            except Exception as exc:
                log.warning("  Failed %s: %s", org, exc)

    log.info("Fetch done — %d repos total", total_fetched)

    # --- Fetch READMEs (only for repos about to be classified) ---
    to_read = get_missing_readme(limit=500)
    log.info("Fetching READMEs for %d unclassified repos...", len(to_read))
    for repo in to_read:
        text = fetch_readme(repo["org"], repo["name"], headers)
        if text:
            update_readme(repo["id"], text)
    log.info("README fetch done")

    # --- Classify ---
    log.info("Classifying unclassified repos...")
    classify_stats = classify_batch(client, limit=500, gh_headers=headers)

    # --- Embed ---
    log.info("Embedding unembedded repos...")
    n_embedded = embed_and_store()
    log.info("Embedded %d repos", n_embedded)

    # --- Cluster ---
    log.info("Clustering embedded repos...")
    n_clusters = cluster_repos()
    log.info("Assigned %d clusters", n_clusters)

    # --- Detect AI providers (ai_ml repos) ---
    ai_repos = get_ai_ml_repos()
    log.info("Detecting AI providers for %d ai_ml repos...", len(ai_repos))
    for repo in ai_repos:
        providers = detect_ai_providers(
            repo["org"], repo["name"], headers,
            readme_text=repo.get("readme_text"),
        )
        update_ai_providers(repo["id"], providers)
        log.info("  %s/%s → %s", repo["org"], repo["name"], providers)

    # --- Catch LLM misses: scan non-ai_ml repos for AI SDK usage ---
    other_repos = get_undetected_classified()
    log.info("Scanning %d non-ai_ml repos for AI SDK usage...", len(other_repos))
    reclassified = 0
    for repo in other_repos:
        providers = detect_ai_providers(
            repo["org"], repo["name"], headers,
            readme_text=repo.get("readme_text"),
        )
        update_ai_providers(repo["id"], providers)
        any_ai = (providers.get("frontier") or providers.get("open_weight")
                  or providers.get("frameworks"))
        if any_ai:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE repos SET domain='ai_ml', llm_confidence=1.0 WHERE id=?",
                    (repo["id"],)
                )
            reclassified += 1
            log.info("  Reclassified %s → ai_ml (detected: %s)",
                     repo["id"], providers)
    log.info("Reclassified %d repos to ai_ml via dependency detection", reclassified)

    # --- Re-scan existing empty-provider repos using stored README text ---
    empty_repos = get_ai_ml_empty_providers()
    log.info("Re-scanning %d repos with empty provider results via README text...",
             len(empty_repos))
    text_hits = 0
    for repo in empty_repos:
        providers = detect_from_text(repo["readme_text"])
        any_found = (providers.get("frontier") or providers.get("open_weight")
                     or providers.get("frameworks"))
        if any_found:
            update_ai_providers(repo["id"], providers)
            text_hits += 1
            log.info("  Text-scan hit: %s → %s", repo["id"], providers)
    log.info("Text-scan upgraded %d previously-empty repos", text_hits)

    log_pipeline_run({
        "repos_fetched":    total_fetched,
        **classify_stats,
    })
    log.info("Pipeline complete.")


if __name__ == "__main__":
    run()
