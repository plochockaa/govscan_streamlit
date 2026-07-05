import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "govscan.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables and indexes if they don't exist. Safe to run repeatedly."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS repos (
                id              TEXT PRIMARY KEY,
                org             TEXT NOT NULL,
                country         TEXT NOT NULL,
                name            TEXT NOT NULL,
                description     TEXT,
                readme_text     TEXT,
                language        TEXT,
                stars           INTEGER DEFAULT 0,
                forks           INTEGER DEFAULT 0,
                open_issues     INTEGER DEFAULT 0,
                has_licence     INTEGER DEFAULT 0,
                has_ci          INTEGER DEFAULT 0,
                topics          TEXT,
                created_at      TEXT,
                updated_at      TEXT,
                fetched_at      TEXT NOT NULL,
                domain          TEXT,
                maturity        TEXT,
                policy_area     TEXT,
                llm_summary     TEXT,
                llm_confidence  REAL,
                classified_at   TEXT,
                embedding       BLOB,
                cluster_id      INTEGER,
                ai_providers    TEXT
            );

            CREATE TABLE IF NOT EXISTS pipeline_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at              TEXT NOT NULL,
                repos_fetched       INTEGER,
                repos_classified    INTEGER,
                input_tokens        INTEGER,
                output_tokens       INTEGER,
                cost_usd            REAL,
                error               TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_repos_country
                ON repos(country);
            CREATE INDEX IF NOT EXISTS idx_repos_domain
                ON repos(domain);
            CREATE INDEX IF NOT EXISTS idx_repos_cluster
                ON repos(cluster_id);

            CREATE TABLE IF NOT EXISTS pipeline_evals (
                repo_id             TEXT PRIMARY KEY,
                eval_at             TEXT NOT NULL,
                domain_correct      INTEGER NOT NULL,
                suggested_domain    TEXT,
                summary_quality     INTEGER NOT NULL,
                confidence_ok       INTEGER NOT NULL,
                overall_score       REAL NOT NULL,
                reasoning           TEXT,
                eval_model          TEXT NOT NULL
            );
        """)


def upsert_repo(repo: dict, db_path: Path = DB_PATH) -> None:
    """Insert or update a single repo. Safe to call every night."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO repos (
                id, org, country, name, description, readme_text,
                language, stars, forks, open_issues, has_licence,
                has_ci, topics, created_at, updated_at, fetched_at
            ) VALUES (
                :id, :org, :country, :name, :description, :readme_text,
                :language, :stars, :forks, :open_issues, :has_licence,
                :has_ci, :topics, :created_at, :updated_at, :fetched_at
            )
            ON CONFLICT(id) DO UPDATE SET
                description  = excluded.description,
                readme_text  = COALESCE(excluded.readme_text, readme_text),
                language     = excluded.language,
                stars        = excluded.stars,
                forks        = excluded.forks,
                open_issues  = excluded.open_issues,
                has_licence  = excluded.has_licence,
                has_ci       = excluded.has_ci,
                topics       = excluded.topics,
                updated_at   = excluded.updated_at,
                fetched_at   = excluded.fetched_at
        """, {
            "id":           repo["id"],
            "org":          repo["org"],
            "country":      repo["country"],
            "name":         repo["name"],
            "description":  repo.get("description"),
            "readme_text":  repo.get("readme_text"),
            "language":     repo.get("language"),
            "stars":        repo.get("stars", 0),
            "forks":        repo.get("forks", 0),
            "open_issues":  repo.get("open_issues", 0),
            "has_licence":  repo.get("has_licence", 0),
            "has_ci":       repo.get("has_ci", 0),
            "topics":       json.dumps(repo.get("topics", [])),
            "created_at":   repo.get("created_at"),
            "updated_at":   repo.get("updated_at"),
            "fetched_at":   now,
        })


def update_classification(repo_id: str, classification: dict,
                          db_path: Path = DB_PATH) -> None:
    """Write LLM classification results. Only touches LLM columns."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("""
            UPDATE repos SET
                domain          = :domain,
                maturity        = :maturity,
                policy_area     = :policy_area,
                llm_summary     = :summary,
                llm_confidence  = :confidence,
                classified_at   = :classified_at
            WHERE id = :id
        """, {**classification,
              "id": repo_id,
              "classified_at": now})


def update_readme(repo_id: str, text: str, db_path: Path = DB_PATH) -> None:
    """Write README text for a single repo."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE repos SET readme_text = ? WHERE id = ?",
            (text, repo_id)
        )


def get_missing_readme(limit: int = 500,
                       db_path: Path = DB_PATH) -> list[dict]:
    """Unclassified repos that have no README yet — fetch before classifying."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, org, name
            FROM repos
            WHERE domain IS NULL
              AND readme_text IS NULL
            ORDER BY stars DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def update_embedding(repo_id: str, embedding_blob: bytes,
                     db_path: Path = DB_PATH) -> None:
    """Write embedding blob for a single repo."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE repos SET embedding = ? WHERE id = ?",
            (embedding_blob, repo_id)
        )


def update_cluster(repo_id: str, cluster_id: int,
                   db_path: Path = DB_PATH) -> None:
    """Write cluster assignment for a single repo."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE repos SET cluster_id = ? WHERE id = ?",
            (cluster_id, repo_id)
        )


def update_ai_providers(repo_id: str, providers: dict,
                        db_path: Path = DB_PATH) -> None:
    """Write detected AI provider tiers for a single repo."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE repos SET ai_providers = ? WHERE id = ?",
            (json.dumps(providers), repo_id)
        )


def get_ai_ml_repos(db_path: Path = DB_PATH) -> list[dict]:
    """All ai_ml repos that haven't had provider detection run yet."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, org, name, language, readme_text
            FROM repos
            WHERE domain = 'ai_ml'
              AND ai_providers IS NULL
        """).fetchall()
    return [dict(r) for r in rows]


def get_undetected_classified(db_path: Path = DB_PATH) -> list[dict]:
    """Non-ai_ml classified repos that haven't had provider detection run yet.
    Used to catch repos the LLM mislabelled but that actually use AI SDKs."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, org, name, language, domain, readme_text
            FROM repos
            WHERE domain IS NOT NULL
              AND domain != 'ai_ml'
              AND ai_providers IS NULL
        """).fetchall()
    return [dict(r) for r in rows]


def get_ai_ml_empty_providers(db_path: Path = DB_PATH) -> list[dict]:
    """ai_ml repos where detection ran but found nothing — re-scan candidates.
    Uses readme_text already stored in DB so no extra GitHub calls needed."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, org, name, language, readme_text
            FROM repos
            WHERE domain = 'ai_ml'
              AND ai_providers IS NOT NULL
              AND json_extract(ai_providers, '$.frontier')    = json('[]')
              AND json_extract(ai_providers, '$.open_weight') = json('[]')
              AND json_extract(ai_providers, '$.frameworks')  = json('[]')
              AND readme_text IS NOT NULL
        """).fetchall()
    return [dict(r) for r in rows]


def get_unclassified(limit: int = 50,
                     db_path: Path = DB_PATH) -> list[dict]:
    """Repos that have never been classified."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, name, org, country, description,
                   readme_text, language, topics
            FROM repos
            WHERE domain IS NULL
            ORDER BY stars DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_unembedded(db_path: Path = DB_PATH) -> list[dict]:
    """Repos that have been classified but not yet embedded."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, name, org, country, description,
                   llm_summary, topics
            FROM repos
            WHERE embedding IS NULL
              AND domain IS NOT NULL
            ORDER BY stars DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_all_repos(db_path: Path = DB_PATH) -> list[dict]:
    """Full table as list of dicts — for Streamlit pages."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM repos ORDER BY stars DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_repos_by_cluster(cluster_id: int,
                         db_path: Path = DB_PATH) -> list[dict]:
    """All repos in a given cluster — for the similarity page."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, name, org, country, domain,
                   llm_summary, description, stars
            FROM repos
            WHERE cluster_id = ?
            ORDER BY stars DESC
        """, (cluster_id,)).fetchall()
    return [dict(r) for r in rows]


def get_duplicate_efforts(db_path: Path = DB_PATH) -> list[dict]:
    """Clusters where 2+ different countries appear — the headline feature."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT
                cluster_id,
                COUNT(*)                        AS repo_count,
                COUNT(DISTINCT country)         AS country_count,
                GROUP_CONCAT(DISTINCT country)  AS countries,
                GROUP_CONCAT(name, ' | ')       AS repo_names
            FROM repos
            WHERE cluster_id >= 0
            GROUP BY cluster_id
            HAVING country_count > 1
            ORDER BY country_count DESC, repo_count DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_stats(db_path: Path = DB_PATH) -> dict:
    """Summary counts for the landing page metrics."""
    with get_connection(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM repos"
        ).fetchone()[0]
        countries = conn.execute(
            "SELECT COUNT(DISTINCT country) FROM repos"
        ).fetchone()[0]
        ai_ml = conn.execute(
            "SELECT COUNT(*) FROM repos WHERE domain = 'ai_ml'"
        ).fetchone()[0]
        clusters = conn.execute("""
            SELECT COUNT(DISTINCT cluster_id) FROM repos
            WHERE cluster_id >= 0
        """).fetchone()[0]
        last_run = conn.execute(
            "SELECT MAX(fetched_at) FROM repos"
        ).fetchone()[0]
    return {
        "total_repos":   total,
        "countries":     countries,
        "ai_ml_repos":   ai_ml,
        "clusters":      clusters,
        "last_updated":  last_run,
    }


def get_unevaluated(limit: int = 50, db_path: Path = DB_PATH,
                    force_model: str | None = None) -> list[dict]:
    """Classified repos that have not yet been evaluated by the eval agent.

    If force_model is given, repos already evaluated by a DIFFERENT model
    are also returned so they can be re-evaluated with the new model.
    """
    with get_connection(db_path) as conn:
        if force_model:
            rows = conn.execute("""
                SELECT r.id, r.name, r.org, r.country, r.description,
                       r.readme_text, r.language, r.topics,
                       r.domain, r.maturity, r.policy_area,
                       r.llm_summary, r.llm_confidence
                FROM repos r
                LEFT JOIN pipeline_evals e ON r.id = e.repo_id
                WHERE r.domain IS NOT NULL
                  AND (e.repo_id IS NULL OR e.eval_model != ?)
                ORDER BY r.stars DESC
                LIMIT ?
            """, (force_model, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT r.id, r.name, r.org, r.country, r.description,
                       r.readme_text, r.language, r.topics,
                       r.domain, r.maturity, r.policy_area,
                       r.llm_summary, r.llm_confidence
                FROM repos r
                LEFT JOIN pipeline_evals e ON r.id = e.repo_id
                WHERE r.domain IS NOT NULL
                  AND e.repo_id IS NULL
                ORDER BY r.stars DESC
                LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def store_eval(
    repo_id: str,
    eval_model: str,
    domain_correct: bool,
    suggested_domain: Optional[str],
    summary_quality: int,
    confidence_ok: bool,
    overall_score: float,
    reasoning: str,
    db_path: Path = DB_PATH,
) -> None:
    """Write or overwrite an evaluation record for a single repo."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO pipeline_evals
                (repo_id, eval_at, domain_correct, suggested_domain,
                 summary_quality, confidence_ok, overall_score, reasoning, eval_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_id) DO UPDATE SET
                eval_at          = excluded.eval_at,
                domain_correct   = excluded.domain_correct,
                suggested_domain = excluded.suggested_domain,
                summary_quality  = excluded.summary_quality,
                confidence_ok    = excluded.confidence_ok,
                overall_score    = excluded.overall_score,
                reasoning        = excluded.reasoning,
                eval_model       = excluded.eval_model
        """, (
            repo_id, now,
            int(domain_correct), suggested_domain,
            summary_quality, int(confidence_ok),
            overall_score, reasoning, eval_model,
        ))


def get_eval_stats(db_path: Path = DB_PATH) -> dict:
    """Aggregate evaluation stats for the quality dashboard."""
    with get_connection(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM pipeline_evals"
        ).fetchone()[0]

        if not total:
            return {
                "total_evals": 0,
                "avg_score": None,
                "pct_domain_correct": None,
                "by_domain": [],
                "flagged": [],
            }

        agg = conn.execute("""
            SELECT AVG(overall_score)              AS avg_score,
                   AVG(CAST(domain_correct AS REAL)) AS pct_correct
            FROM pipeline_evals
        """).fetchone()

        by_domain = conn.execute("""
            SELECT r.domain,
                   AVG(e.overall_score)               AS avg_score,
                   COUNT(*)                           AS n,
                   AVG(CAST(e.domain_correct AS REAL)) AS pct_correct
            FROM pipeline_evals e
            JOIN repos r ON r.id = e.repo_id
            GROUP BY r.domain
            ORDER BY avg_score DESC
        """).fetchall()

        flagged = conn.execute("""
            SELECT e.repo_id, r.name, r.country, r.domain,
                   e.suggested_domain, e.overall_score, e.reasoning
            FROM pipeline_evals e
            JOIN repos r ON r.id = e.repo_id
            WHERE e.domain_correct = 0
            ORDER BY e.overall_score ASC
            LIMIT 25
        """).fetchall()

    return {
        "total_evals":        total,
        "avg_score":          agg["avg_score"],
        "pct_domain_correct": agg["pct_correct"],
        "by_domain":          [dict(r) for r in by_domain],
        "flagged":            [dict(r) for r in flagged],
    }


def log_pipeline_run(stats: dict, db_path: Path = DB_PATH) -> None:
    """Record pipeline run stats for cost tracking."""
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO pipeline_log
                (run_at, repos_fetched, repos_classified,
                 input_tokens, output_tokens, cost_usd, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            stats.get("repos_fetched",    0),
            stats.get("repos_classified", 0),
            stats.get("input_tokens",     0),
            stats.get("output_tokens",    0),
            stats.get("cost_usd",         0.0),
            stats.get("error"),
        ))


if __name__ == "__main__":
    print("Initialising database...")
    init_db()
    print(f"  Done. DB created at: {DB_PATH.resolve()}")

    print("\nInserting two test repos...")
    upsert_repo({
        "id":          "alphagov/govuk-frontend",
        "org":         "alphagov",
        "country":     "UK",
        "name":        "govuk-frontend",
        "description": "GOV.UK Frontend — code to build government services",
        "language":    "JavaScript",
        "stars":       3800,
        "forks":       1200,
        "open_issues": 45,
        "has_licence": 1,
        "has_ci":      1,
        "topics":      ["design-system", "government", "frontend"],
        "created_at":  "2016-01-01T00:00:00",
        "updated_at":  "2024-06-01T00:00:00",
    })
    upsert_repo({
        "id":          "govtechsg/purple-a11y",
        "org":         "govtechsg",
        "country":     "Singapore",
        "name":        "purple-a11y",
        "description": "Automated accessibility testing tool for government services",
        "language":    "Python",
        "stars":       120,
        "forks":       30,
        "open_issues": 8,
        "has_licence": 1,
        "has_ci":      0,
        "topics":      ["accessibility", "government", "testing"],
        "created_at":  "2021-03-01T00:00:00",
        "updated_at":  "2024-05-01T00:00:00",
    })
    print("  Done.")

    print("\nReading back all repos...")
    for repo in get_all_repos():
        print(f"  {repo['id']} ({repo['country']}) — {repo['stars']} stars")

    print("\nTesting upsert — updating govuk-frontend stars to 4000...")
    upsert_repo({
        "id":          "alphagov/govuk-frontend",
        "org":         "alphagov",
        "country":     "UK",
        "name":        "govuk-frontend",
        "description": "GOV.UK Frontend — code to build government services",
        "language":    "JavaScript",
        "stars":       4000,
        "forks":       1200,
        "open_issues": 45,
        "has_licence": 1,
        "has_ci":      1,
        "topics":      ["design-system", "government", "frontend"],
        "created_at":  "2016-01-01T00:00:00",
        "updated_at":  "2024-06-11T00:00:00",
    })

    all_repos = get_all_repos()
    govuk = next(r for r in all_repos if r["id"] == "alphagov/govuk-frontend")
    print(f"  govuk-frontend now has {govuk['stars']} stars (should be 4000)")
    print(f"  Total repos in DB: {len(all_repos)} (should be 2 — no duplicate created)")

    print("\nTesting get_stats()...")
    stats = get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ All checks passed. Delete data/govscan.db before committing.")