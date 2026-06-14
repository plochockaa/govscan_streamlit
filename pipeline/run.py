import logging
import os

from mistralai.client.sdk import Mistral

from config import ORGS
from pipeline.classify import classify_batch
from pipeline.embed import embed_and_store
from pipeline.fetch import fetch_org_repos
from pipeline.store import init_db, upsert_repo

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
    mistral_key = os.environ["MISTRAL_API_KEY"]

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github+json",
    }
    client = Mistral(api_key=mistral_key)

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

    # --- Classify ---
    log.info("Classifying unclassified repos...")
    classify_batch(client, limit=500)

    # --- Embed ---
    log.info("Embedding unembedded repos...")
    n_embedded = embed_and_store()
    log.info("Embedded %d repos", n_embedded)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    run()
