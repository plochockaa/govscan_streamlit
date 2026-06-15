"""
Scans a repo's dependency files for AI SDK references.
Only makes 3-4 GitHub API calls per repo, so safe to run on all ai_ml repos.
"""
import base64
import logging
import re
import time

import requests

log = logging.getLogger(__name__)

# fmt: off
FRONTIER: dict[str, list[str]] = {
    "openai":     ["openai", "@openai/openai", "azure-openai"],
    "anthropic":  ["anthropic", "@anthropic-ai/sdk"],
    "google":     ["google-generativeai", "google-cloud-aiplatform",
                   "vertexai", "@google-ai/generativelanguage"],
    "aws-bedrock":["boto3"],           # imprecise but most gov AWS AI use goes through bedrock
}

OPEN_WEIGHT: dict[str, list[str]] = {
    "mistral":      ["mistralai"],
    "cohere":       ["cohere"],
    "huggingface":  ["transformers", "huggingface-hub", "huggingface_hub",
                     "diffusers", "sentence-transformers", "@huggingface/inference"],
    "ollama":       ["ollama"],
    "llama":        ["llama-cpp-python", "llama_cpp"],
}

FRAMEWORKS: dict[str, list[str]] = {
    "langchain": ["langchain", "langchain-openai", "langchain-anthropic",
                  "langchain-community", "langchain-google-genai"],
    "llamaindex":["llama-index", "llama_index", "llama-index-core"],
    "litellm":   ["litellm"],
    "haystack":  ["haystack-ai", "farm-haystack"],
    "semantic-kernel": ["semantic-kernel"],
}
# fmt: on

_DEP_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "Pipfile",
    "setup.cfg",
]

_ALL_PKGS: dict[str, str] = {}
for _cat, _pkgs in {**FRONTIER, **OPEN_WEIGHT, **FRAMEWORKS}.items():
    for _p in _pkgs:
        _ALL_PKGS[_p.lower()] = _cat


def _fetch_text(org: str, name: str, path: str, headers: dict) -> str | None:
    url = f"https://api.github.com/repos/{org}/{name}/contents/{path}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    data = r.json()
    if isinstance(data, dict) and data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return None


def detect_ai_providers(org: str, name: str, headers: dict) -> dict[str, list[str]]:
    """
    Fetch dependency files and return detected AI providers grouped by tier.

    Returns:
        {
          "frontier":   ["openai", ...],
          "open_weight": ["mistral", ...],
          "frameworks": ["langchain", ...],
        }
    """
    found: dict[str, set] = {"frontier": set(), "open_weight": set(), "frameworks": set()}

    tier_map = {
        **{k: "frontier"    for k in FRONTIER},
        **{k: "open_weight" for k in OPEN_WEIGHT},
        **{k: "frameworks"  for k in FRAMEWORKS},
    }

    for dep_file in _DEP_FILES:
        content = _fetch_text(org, name, dep_file, headers)
        if not content:
            continue
        content_lower = content.lower()
        for pkg, provider in _ALL_PKGS.items():
            # match package name with word boundary, tolerating - vs _ variation
            pattern = r"[\s\[\"\'=<>!~^;,]" + re.escape(pkg.replace("-", "[-_]")) + r"[\s\[\"\'=<>!~^;,\n]"
            if re.search(pattern, content_lower):
                tier = tier_map[provider]
                found[tier].add(provider)
        time.sleep(0.1)   # be gentle with the API

    return {k: sorted(v) for k, v in found.items()}
