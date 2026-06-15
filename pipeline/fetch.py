import time
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_page(url: str, headers: dict) -> requests.Response:
    """Fetch a single page with retry on failure."""
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp


def fetch_all_pages(url: str, headers: dict) -> list[dict]:
    """Paginate through all pages of a GitHub API endpoint."""
    results = []
    while url:
        resp = fetch_page(url, headers)
        results.extend(resp.json())
        url = resp.links.get("next", {}).get("url")
    return results


def check_rate_limit(headers: dict) -> None:
    """Pause if approaching GitHub rate limit."""
    try:
        r = requests.get("https://api.github.com/rate_limit",
                         headers=headers).json()
        remaining = r.get("resources", {}).get("core", {}).get("remaining", 999)
        reset_time = r.get("resources", {}).get("core", {}).get("reset", 0)
    except Exception:
        logger.warning("Could not check rate limit — skipping")
        return

    if remaining < 100:
        sleep_secs = max(0, reset_time - time.time()) + 5
        logger.warning(f"Rate limit low ({remaining}). Sleeping {sleep_secs:.0f}s")
        time.sleep(sleep_secs)


def fetch_org_repos(org: str, headers: dict) -> list[dict]:
    """Fetch all repos for an org across all pages."""
    check_rate_limit(headers)
    url = f"https://api.github.com/orgs/{org}/repos?per_page=100"
    return fetch_all_pages(url, headers)


def fetch_readme(org: str, name: str, headers: dict,
                 max_chars: int = 3000) -> str | None:
    """Fetch and decode the README for a single repo. Returns None on 404 or error."""
    import base64
    url = f"https://api.github.com/repos/{org}/{name}/readme"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        content = resp.json().get("content", "")
        text = base64.b64decode(content).decode("utf-8", errors="replace")
        return text[:max_chars]
    except Exception as exc:
        logger.warning("README fetch failed for %s/%s: %s", org, name, exc)
        return None