import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline.classify import (
    ClassificationResult,
    Domain,
    _build_user_msg,
    _log_batch,
    classify_batch,
    classify_repo,
)

VALID_RESULT = {
    "domain": "ai_ml",
    "maturity": "active",
    "policy_area": "cross_cutting",
    "summary": "A machine learning toolkit for government services.",
    "confidence": 0.85,
}

REPO = {
    "id": "alphagov/govuk-ml",
    "name": "govuk-ml",
    "description": "ML tools for GOV.UK",
    "language": "Python",
    "topics": ["machine-learning", "government"],
    "readme_text": "This repo contains ML tools." * 10,
}


def make_client(result: dict = VALID_RESULT, prompt_tokens=100, completion_tokens=50):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = json.dumps(result)

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage

    client = MagicMock()
    client.chat.complete.return_value = response
    return client


class TestBuildUserMsg:
    def test_includes_all_fields(self):
        msg = _build_user_msg(REPO)
        assert "govuk-ml" in msg
        assert "ML tools for GOV.UK" in msg
        assert "Python" in msg
        assert "machine-learning" in msg

    def test_readme_truncated_to_800_chars(self):
        repo = {**REPO, "readme_text": "x" * 1200}
        msg = _build_user_msg(repo)
        readme_part = msg.split("readme: ")[1]
        assert len(readme_part) == 800

    def test_missing_optional_fields_use_defaults(self):
        repo = {"name": "bare-repo"}
        msg = _build_user_msg(repo)
        assert "description: none" in msg
        assert "language: unknown" in msg
        assert "topics: none" in msg


class TestClassificationResult:
    def test_valid_result(self):
        result = ClassificationResult(**VALID_RESULT)
        assert result.domain == Domain.AI_ML
        assert result.confidence == 0.85

    def test_rejects_unknown_domain(self):
        with pytest.raises(ValidationError):
            ClassificationResult(**{**VALID_RESULT, "domain": "not_a_domain"})

    def test_rejects_unknown_maturity(self):
        with pytest.raises(ValidationError):
            ClassificationResult(**{**VALID_RESULT, "maturity": "legacy"})

    def test_rejects_unknown_policy_area(self):
        with pytest.raises(ValidationError):
            ClassificationResult(**{**VALID_RESULT, "policy_area": "defence"})


class TestClassifyRepo:
    def test_returns_classification_result(self):
        client = make_client()
        result = classify_repo(REPO, client)
        assert isinstance(result, ClassificationResult)
        assert result.domain == Domain.AI_ML

    def test_passes_correct_model_to_api(self):
        client = make_client()
        classify_repo(REPO, client)
        call_kwargs = client.chat.complete.call_args.kwargs
        assert call_kwargs["model"] == "open-mistral-nemo"

    def test_uses_json_response_format(self):
        client = make_client()
        classify_repo(REPO, client)
        call_kwargs = client.chat.complete.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestLogBatch:
    def test_writes_jsonl_entry(self, tmp_path):
        log_path = tmp_path / "pipeline_log.jsonl"
        with patch("pipeline.classify._LOG_PATH", log_path):
            _log_batch(n_repos=5, input_tokens=1000, output_tokens=500, tool_calls=0)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["repos_classified"] == 5
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert "run_at" in entry
        assert "cost_usd" in entry

    def test_appends_on_multiple_calls(self, tmp_path):
        log_path = tmp_path / "pipeline_log.jsonl"
        with patch("pipeline.classify._LOG_PATH", log_path):
            _log_batch(3, 500, 200, tool_calls=0)
            _log_batch(2, 300, 100, tool_calls=1)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_cost_calculation(self, tmp_path):
        log_path = tmp_path / "pipeline_log.jsonl"
        with patch("pipeline.classify._LOG_PATH", log_path):
            # 1M input tokens at $0.15 + 1M output tokens at $0.15 = $0.30
            _log_batch(1, 1_000_000, 1_000_000, tool_calls=0)

        entry = json.loads(log_path.read_text())
        assert entry["cost_usd"] == pytest.approx(0.30)


class TestClassifyBatch:
    @patch("pipeline.classify.update_classification")
    @patch("pipeline.classify.get_unclassified")
    def test_classifies_all_repos(self, mock_get, mock_update):
        mock_get.return_value = [REPO, {**REPO, "id": "alphagov/repo-b", "name": "repo-b"}]
        client = make_client()

        with patch("pipeline.classify._LOG_PATH", Path("/dev/null")):
            classify_batch(client, limit=10)

        assert client.chat.complete.call_count == 2
        assert mock_update.call_count == 2

    @patch("pipeline.classify.update_classification")
    @patch("pipeline.classify.get_unclassified")
    def test_no_log_written_when_no_repos(self, mock_get, mock_update, tmp_path):
        mock_get.return_value = []
        log_path = tmp_path / "pipeline_log.jsonl"
        client = make_client()

        with patch("pipeline.classify._LOG_PATH", log_path):
            classify_batch(client)

        assert not log_path.exists()

    @patch("pipeline.classify.update_classification")
    @patch("pipeline.classify.get_unclassified")
    def test_accumulates_token_counts(self, mock_get, mock_update, tmp_path):
        mock_get.return_value = [REPO, {**REPO, "id": "alphagov/repo-b", "name": "repo-b"}]
        log_path = tmp_path / "pipeline_log.jsonl"
        client = make_client(prompt_tokens=100, completion_tokens=50)

        with patch("pipeline.classify._LOG_PATH", log_path):
            classify_batch(client)

        entry = json.loads(log_path.read_text())
        assert entry["input_tokens"] == 200
        assert entry["output_tokens"] == 100
