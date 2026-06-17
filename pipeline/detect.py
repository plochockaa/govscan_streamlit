"""
Scans a repo's dependency files and README text for AI SDK references.
Dep-file scan makes 3-4 GitHub API calls per repo; README scan uses
already-stored text (zero extra calls).
"""
import base64
import logging
import re
import time

import requests

log = logging.getLogger(__name__)

# fmt: off
FRONTIER: dict[str, list[str]] = {
    "openai":     ["openai", "@openai/openai", "azure-openai", "tiktoken"],
    "anthropic":  ["anthropic", "@anthropic-ai/sdk"],
    "google":     ["google-generativeai", "google-cloud-aiplatform",
                   "vertexai", "@google-ai/generativelanguage"],
    "aws-bedrock":["boto3"],           # imprecise but most gov AWS AI use goes through bedrock
}

OPEN_WEIGHT: dict[str, list[str]] = {
    "mistral":      ["mistralai"],
    "cohere":       ["cohere"],
    "huggingface":  ["transformers", "huggingface-hub", "huggingface_hub",
                     "diffusers", "sentence-transformers", "@huggingface/inference",
                     "trl", "peft", "accelerate", "bitsandbytes"],
    "ollama":       ["ollama"],
    "llama":        ["llama-cpp-python", "llama_cpp", "ctransformers"],
    "groq":         ["groq"],
    "vllm":         ["vllm"],
    "together":     ["together"],
}

FRAMEWORKS: dict[str, list[str]] = {
    "langchain": ["langchain", "langchain-openai", "langchain-anthropic",
                  "langchain-community", "langchain-google-genai"],
    "llamaindex":["llama-index", "llama_index", "llama-index-core"],
    "litellm":   ["litellm"],
    "haystack":  ["haystack-ai", "farm-haystack"],
    "semantic-kernel": ["semantic-kernel"],
    "dspy":      ["dspy-ai", "dspy"],
    "autogen":   ["pyautogen", "autogen-agentchat"],
    "instructor":["instructor"],
    "crewai":    ["crewai"],
    "outlines":  ["outlines"],
}
# fmt: on

_DEP_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "Pipfile",
    "setup.cfg",
    "setup.py",
    "environment.yml",
    "conda.yml",
]

# Language-specific dep files that use different package declaration syntax.
# Each entry is (provider_name, regex_pattern); patterns are case-insensitive.
_LANG_DEP_FILES: dict[str, list[tuple[str, str]]] = {
    "go.mod": [
        ("openai",      r"go-openai"),
        ("anthropic",   r"anthropic-sdk-go"),
        ("google",      r"generative-ai-go"),
        ("ollama",      r"ollama/ollama"),
        ("groq",        r"groq-go"),
    ],
    "Gemfile": [
        ("openai",      r"gem\s+['\"](?:ruby-openai|openai)['\"]"),
        ("anthropic",   r"gem\s+['\"]anthropic['\"]"),
        ("ollama",      r"gem\s+['\"]ollama"),
        ("huggingface", r"gem\s+['\"](?:hugging.?face|ruby-huggingface)['\"]"),
    ],
    "composer.json": [
        ("openai",      r"openai-php|openai/openai"),
        ("anthropic",   r"anthropics/anthropic"),
        ("huggingface", r"huggingface"),
    ],
}

# Patterns scanned against free text (README, descriptions).
# Each key must be a provider that exists in FRONTIER / OPEN_WEIGHT / FRAMEWORKS.
# Patterns are tried with re.IGNORECASE; first match wins for that provider.
_TEXT_SIGNALS: dict[str, list[str]] = {
    "openai":      [r"OPENAI_API_KEY", r"api\.openai\.com", r"\bgpt-4\b",
                    r"\bgpt-3\.5-turbo\b", r"\bgpt-35\b"],
    "anthropic":   [r"ANTHROPIC_API_KEY", r"api\.anthropic\.com",
                    r"\bclaude-[23][-.]",
                    r"\banthropicbedrock\b", r"anthropic\.claude"],   # Bedrock model ID
    "google":      [r"GOOGLE_API_KEY", r"GEMINI_API_KEY",
                    r"\bgemini-(?:pro|1\.[05])\b",
                    r"generativelanguage\.googleapis\.com"],
    "aws-bedrock": [r"AWS_BEDROCK", r"bedrock\.amazonaws\.com",
                    r"bedrock-runtime"],
    "huggingface": [r"\bHF_TOKEN\b", r"HUGGINGFACE_TOKEN",
                    r"huggingface\.co/\w", r"\bhf\.co/\w"],
    "ollama":      [r"\bollama\b"],
    "llama":       [r"\bllama-?[23]\b", r"\bllama2\b", r"\bllama3\b",
                    r"\bmeta-llama\b",
                    r"\bmeta\.llama"],                                 # Bedrock model ID
    "mistral":     [r"\bmistralai/", r"\bmistral-\d", r"\bmixtral\b",
                    r"\bmistral\.(?:mistral|mixtral)"],                # Bedrock model ID
    "cohere":      [r"\bcohere\.command"],                             # Bedrock model ID
    "groq":        [r"\bgroq\b"],
    "vllm":        [r"\bvllm\b"],
    "together":    [r"TOGETHER_API_KEY", r"api\.together\.xyz", r"together\.ai"],
}

_TIER_MAP: dict[str, str] = {
    **{k: "frontier"    for k in FRONTIER},
    **{k: "open_weight" for k in OPEN_WEIGHT},
    **{k: "frameworks"  for k in FRAMEWORKS},
}

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


def _fetch_root_filenames(org: str, name: str, headers: dict) -> list[str]:
    """List root-level filenames — used to discover requirements*.txt variants."""
    url = f"https://api.github.com/repos/{org}/{name}/contents/"
    try:
        r = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    return [item["path"] for item in data if item.get("type") == "file"]


def detect_from_text(text: str) -> dict[str, list[str]]:
    """
    Scan arbitrary text (README, description) for AI provider signals.
    Returns the same tier-grouped structure as detect_ai_providers.
    Zero network calls — operates on already-stored text.
    """
    found: dict[str, set] = {"frontier": set(), "open_weight": set(), "frameworks": set()}
    for provider, patterns in _TEXT_SIGNALS.items():
        tier = _TIER_MAP.get(provider)
        if not tier:
            continue
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found[tier].add(provider)
                break
    return {k: sorted(v) for k, v in found.items()}


def detect_ai_providers(
    org: str,
    name: str,
    headers: dict,
    readme_text: str | None = None,
) -> dict[str, list[str]]:
    """
    Fetch dependency files and return detected AI providers grouped by tier.
    If readme_text is provided, text-signal results are merged in at no extra cost.

    Returns:
        {
          "frontier":    ["openai", ...],
          "open_weight": ["mistral", ...],
          "frameworks":  ["langchain", ...],
        }
    """
    found: dict[str, set] = {"frontier": set(), "open_weight": set(), "frameworks": set()}

    def _scan_pkg_file(content: str) -> None:
        content_lower = content.lower()
        for pkg, provider in _ALL_PKGS.items():
            pattern = (r"[\s\[\"\'=<>!~^;,]"
                       + re.escape(pkg.replace("-", "[-_]"))
                       + r"[\s\[\"\'=<>!~^;,\n]")
            if re.search(pattern, content_lower):
                found[_TIER_MAP[provider]].add(provider)

    for dep_file in _DEP_FILES:
        content = _fetch_text(org, name, dep_file, headers)
        if content:
            _scan_pkg_file(content)
        time.sleep(0.1)

    # requirements*.txt variants (e.g. requirements-dev.txt, requirements_test.txt)
    root_files = _fetch_root_filenames(org, name, headers)
    time.sleep(0.1)
    for path in root_files:
        if re.match(r"requirements.*\.txt$", path, re.IGNORECASE) and path not in _DEP_FILES:
            content = _fetch_text(org, name, path, headers)
            if content:
                _scan_pkg_file(content)
            time.sleep(0.1)

    # Language-specific dep files (Go, Ruby, PHP)
    for dep_file, sig_patterns in _LANG_DEP_FILES.items():
        content = _fetch_text(org, name, dep_file, headers)
        if not content:
            continue
        for provider, pat in sig_patterns:
            if re.search(pat, content, re.IGNORECASE):
                tier = _TIER_MAP.get(provider)
                if tier:
                    found[tier].add(provider)
        time.sleep(0.1)

    if readme_text:
        for tier, providers in detect_from_text(readme_text).items():
            found[tier].update(providers)

    return {k: sorted(v) for k, v in found.items()}
