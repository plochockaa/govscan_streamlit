import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.store import (
    get_all_repos,
    get_duplicate_efforts,
    get_repos_by_cluster,
    get_stats,
    get_unclassified,
    get_unembedded,
    init_db,
    log_pipeline_run,
    update_classification,
    update_cluster,
    update_embedding,
    upsert_repo,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _repo(overrides: dict | None = None) -> dict:
    base = {
        "id": "org/repo-a",
        "org": "org",
        "country": "UK",
        "name": "repo-a",
        "description": "A test repo",
        "language": "Python",
        "stars": 100,
        "forks": 10,
        "open_issues": 2,
        "has_licence": 1,
        "has_ci": 1,
        "topics": ["gov", "api"],
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    return {**base, **(overrides or {})}


class TestInitDb:
    def test_creates_tables(self, db: Path):
        from pipeline.store import get_connection
        with get_connection(db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "repos" in tables
        assert "pipeline_log" in tables

    def test_idempotent(self, db: Path):
        init_db(db)  # calling twice should not raise
        init_db(db)


class TestUpsertRepo:
    def test_inserts_new_repo(self, db: Path):
        upsert_repo(_repo(), db)
        repos = get_all_repos(db)
        assert len(repos) == 1
        assert repos[0]["id"] == "org/repo-a"

    def test_updates_existing_repo(self, db: Path):
        upsert_repo(_repo(), db)
        upsert_repo(_repo({"stars": 999}), db)
        repos = get_all_repos(db)
        assert len(repos) == 1
        assert repos[0]["stars"] == 999

    def test_preserves_classification_on_update(self, db: Path):
        upsert_repo(_repo(), db)
        update_classification("org/repo-a", {
            "domain": "devtools",
            "maturity": "active",
            "policy_area": "cross_cutting",
            "summary": "A devtool",
            "confidence": 0.9,
        }, db)
        upsert_repo(_repo({"stars": 500}), db)
        repos = get_all_repos(db)
        assert repos[0]["domain"] == "devtools"

    def test_stores_topics_as_json(self, db: Path):
        upsert_repo(_repo({"topics": ["a", "b"]}), db)
        repos = get_all_repos(db)
        assert repos[0]["topics"] == '["a", "b"]'

    def test_inserts_multiple_repos(self, db: Path):
        upsert_repo(_repo({"id": "org/repo-a"}), db)
        upsert_repo(_repo({"id": "org/repo-b"}), db)
        assert len(get_all_repos(db)) == 2


class TestUpdateClassification:
    def test_writes_llm_columns(self, db: Path):
        upsert_repo(_repo(), db)
        update_classification("org/repo-a", {
            "domain": "ai_ml",
            "maturity": "prototype",
            "policy_area": "health",
            "summary": "ML tool for health",
            "confidence": 0.85,
        }, db)
        repos = get_all_repos(db)
        r = repos[0]
        assert r["domain"] == "ai_ml"
        assert r["maturity"] == "prototype"
        assert r["policy_area"] == "health"
        assert r["llm_summary"] == "ML tool for health"
        assert abs(r["llm_confidence"] - 0.85) < 1e-6

    def test_does_not_overwrite_stars(self, db: Path):
        upsert_repo(_repo({"stars": 42}), db)
        update_classification("org/repo-a", {
            "domain": "other",
            "maturity": "archived",
            "policy_area": "unknown",
            "summary": "misc",
            "confidence": 0.5,
        }, db)
        assert get_all_repos(db)[0]["stars"] == 42


class TestUpdateEmbedding:
    def test_writes_and_reads_back_blob(self, db: Path):
        upsert_repo(_repo(), db)
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        update_embedding("org/repo-a", vec.tobytes(), db)

        from pipeline.store import get_connection
        with get_connection(db) as conn:
            blob = conn.execute(
                "SELECT embedding FROM repos WHERE id = 'org/repo-a'"
            ).fetchone()["embedding"]
        result = np.frombuffer(blob, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, vec)


class TestUpdateCluster:
    def test_writes_cluster_id(self, db: Path):
        upsert_repo(_repo(), db)
        update_cluster("org/repo-a", 7, db)
        repos = get_all_repos(db)
        assert repos[0]["cluster_id"] == 7


class TestGetUnclassified:
    def test_returns_repos_without_domain(self, db: Path):
        upsert_repo(_repo({"id": "org/a"}), db)
        upsert_repo(_repo({"id": "org/b"}), db)
        update_classification("org/a", {
            "domain": "devtools", "maturity": "active",
            "policy_area": "cross_cutting", "summary": "x", "confidence": 1.0,
        }, db)
        result = get_unclassified(db_path=db)
        assert len(result) == 1
        assert result[0]["id"] == "org/b"

    def test_respects_limit(self, db: Path):
        for i in range(5):
            upsert_repo(_repo({"id": f"org/repo-{i}"}), db)
        result = get_unclassified(limit=3, db_path=db)
        assert len(result) == 3


class TestGetUnembedded:
    def test_returns_classified_without_embedding(self, db: Path):
        upsert_repo(_repo({"id": "org/a"}), db)
        upsert_repo(_repo({"id": "org/b"}), db)
        for repo_id in ("org/a", "org/b"):
            update_classification(repo_id, {
                "domain": "devtools", "maturity": "active",
                "policy_area": "cross_cutting", "summary": "x", "confidence": 1.0,
            }, db)
        update_embedding("org/a", np.array([1.0], dtype=np.float32).tobytes(), db)

        result = get_unembedded(db_path=db)
        assert len(result) == 1
        assert result[0]["id"] == "org/b"

    def test_excludes_unclassified(self, db: Path):
        upsert_repo(_repo(), db)
        result = get_unembedded(db_path=db)
        assert result == []


class TestGetAllRepos:
    def test_ordered_by_stars_desc(self, db: Path):
        upsert_repo(_repo({"id": "org/low", "stars": 1}), db)
        upsert_repo(_repo({"id": "org/high", "stars": 1000}), db)
        repos = get_all_repos(db)
        assert repos[0]["id"] == "org/high"
        assert repos[1]["id"] == "org/low"

    def test_empty_db_returns_empty_list(self, db: Path):
        assert get_all_repos(db) == []


class TestGetReposByCluster:
    def test_returns_only_matching_cluster(self, db: Path):
        upsert_repo(_repo({"id": "org/a"}), db)
        upsert_repo(_repo({"id": "org/b"}), db)
        update_cluster("org/a", 1, db)
        update_cluster("org/b", 2, db)

        result = get_repos_by_cluster(1, db)
        assert len(result) == 1
        assert result[0]["id"] == "org/a"

    def test_returns_empty_for_unknown_cluster(self, db: Path):
        assert get_repos_by_cluster(99, db) == []


class TestGetDuplicateEfforts:
    def test_finds_multi_country_clusters(self, db: Path):
        upsert_repo(_repo({"id": "org/uk", "country": "UK"}), db)
        upsert_repo(_repo({"id": "org/sg", "country": "Singapore"}), db)
        update_cluster("org/uk", 1, db)
        update_cluster("org/sg", 1, db)

        result = get_duplicate_efforts(db)
        assert len(result) == 1
        assert result[0]["country_count"] == 2

    def test_excludes_single_country_clusters(self, db: Path):
        upsert_repo(_repo({"id": "org/uk1", "country": "UK"}), db)
        upsert_repo(_repo({"id": "org/uk2", "country": "UK"}), db)
        update_cluster("org/uk1", 1, db)
        update_cluster("org/uk2", 1, db)

        assert get_duplicate_efforts(db) == []

    def test_excludes_unclustered_repos(self, db: Path):
        upsert_repo(_repo({"id": "org/uk", "country": "UK"}), db)
        upsert_repo(_repo({"id": "org/sg", "country": "Singapore"}), db)
        # no cluster assigned — cluster_id is NULL, not >= 0
        assert get_duplicate_efforts(db) == []


class TestGetStats:
    def test_counts_total_and_countries(self, db: Path):
        upsert_repo(_repo({"id": "org/a", "country": "UK"}), db)
        upsert_repo(_repo({"id": "org/b", "country": "Singapore"}), db)
        stats = get_stats(db)
        assert stats["total_repos"] == 2
        assert stats["countries"] == 2

    def test_counts_ai_ml_repos(self, db: Path):
        upsert_repo(_repo({"id": "org/a"}), db)
        upsert_repo(_repo({"id": "org/b"}), db)
        update_classification("org/a", {
            "domain": "ai_ml", "maturity": "active",
            "policy_area": "cross_cutting", "summary": "x", "confidence": 1.0,
        }, db)
        stats = get_stats(db)
        assert stats["ai_ml_repos"] == 1

    def test_counts_clusters(self, db: Path):
        upsert_repo(_repo({"id": "org/a"}), db)
        upsert_repo(_repo({"id": "org/b"}), db)
        update_cluster("org/a", 0, db)
        update_cluster("org/b", 1, db)
        stats = get_stats(db)
        assert stats["clusters"] == 2

    def test_empty_db(self, db: Path):
        stats = get_stats(db)
        assert stats["total_repos"] == 0
        assert stats["countries"] == 0


class TestLogPipelineRun:
    def test_writes_row_to_pipeline_log(self, db: Path):
        log_pipeline_run({
            "repos_fetched": 10,
            "repos_classified": 8,
            "input_tokens": 1000,
            "output_tokens": 200,
            "cost_usd": 0.0005,
        }, db)

        from pipeline.store import get_connection
        with get_connection(db) as conn:
            rows = conn.execute("SELECT * FROM pipeline_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["repos_fetched"] == 10
        assert rows[0]["repos_classified"] == 8

    def test_handles_missing_keys_gracefully(self, db: Path):
        log_pipeline_run({}, db)  # all optional — should not raise
        from pipeline.store import get_connection
        with get_connection(db) as conn:
            rows = conn.execute("SELECT * FROM pipeline_log").fetchall()
        assert len(rows) == 1
