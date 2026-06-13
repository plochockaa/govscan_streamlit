from unittest.mock import MagicMock, patch

import pytest

from pipeline.fetch import check_rate_limit, fetch_all_pages, fetch_org_repos


def make_response(data: list, next_url: str | None = None) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.links = {"next": {"url": next_url}} if next_url else {}
    resp.raise_for_status.return_value = None
    return resp


HEADERS = {"Authorization": "token test", "Accept": "application/vnd.github+json"}


class TestFetchAllPages:
    @patch("pipeline.fetch.fetch_page")
    def test_single_page(self, mock_fetch):
        mock_fetch.return_value = make_response([{"id": 1}, {"id": 2}])
        result = fetch_all_pages("https://api.github.com/orgs/test/repos", HEADERS)
        assert result == [{"id": 1}, {"id": 2}]
        assert mock_fetch.call_count == 1

    @patch("pipeline.fetch.fetch_page")
    def test_multiple_pages(self, mock_fetch):
        mock_fetch.side_effect = [
            make_response([{"id": 1}], next_url="https://api.github.com/orgs/test/repos?page=2"),
            make_response([{"id": 2}], next_url="https://api.github.com/orgs/test/repos?page=3"),
            make_response([{"id": 3}]),
        ]
        result = fetch_all_pages("https://api.github.com/orgs/test/repos", HEADERS)
        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert mock_fetch.call_count == 3

    @patch("pipeline.fetch.fetch_page")
    def test_empty_response(self, mock_fetch):
        mock_fetch.return_value = make_response([])
        result = fetch_all_pages("https://api.github.com/orgs/test/repos", HEADERS)
        assert result == []


class TestCheckRateLimit:
    @patch("pipeline.fetch.requests.get")
    def test_pauses_when_limit_low(self, mock_get):
        import time
        mock_get.return_value.json.return_value = {
            "resources": {"core": {"remaining": 50, "reset": time.time() + 10}}
        }
        with patch("pipeline.fetch.time.sleep") as mock_sleep:
            check_rate_limit(HEADERS)
            mock_sleep.assert_called_once()

    @patch("pipeline.fetch.requests.get")
    def test_no_pause_when_limit_ok(self, mock_get):
        mock_get.return_value.json.return_value = {
            "resources": {"core": {"remaining": 500, "reset": 0}}
        }
        with patch("pipeline.fetch.time.sleep") as mock_sleep:
            check_rate_limit(HEADERS)
            mock_sleep.assert_not_called()

    @patch("pipeline.fetch.requests.get", side_effect=Exception("network error"))
    def test_does_not_raise_on_error(self, _):
        check_rate_limit(HEADERS)  # should not raise


class TestFetchOrgRepos:
    @patch("pipeline.fetch.fetch_all_pages")
    @patch("pipeline.fetch.check_rate_limit")
    def test_constructs_correct_url(self, mock_rate, mock_pages):
        mock_pages.return_value = []
        fetch_org_repos("alphagov", HEADERS)
        called_url = mock_pages.call_args[0][0]
        assert "alphagov" in called_url
        assert "per_page=100" in called_url

    @patch("pipeline.fetch.fetch_all_pages")
    @patch("pipeline.fetch.check_rate_limit")
    def test_returns_repos(self, mock_rate, mock_pages):
        mock_pages.return_value = [{"name": "repo-a"}, {"name": "repo-b"}]
        result = fetch_org_repos("alphagov", HEADERS)
        assert len(result) == 2
        assert result[0]["name"] == "repo-a"
