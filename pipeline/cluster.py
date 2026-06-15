from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from pipeline.store import DB_PATH, get_connection, update_cluster


def _load_embeddings(db_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (repo_ids, matrix) for all embedded repos."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM repos WHERE embedding IS NOT NULL"
        ).fetchall()

    ids = [r["id"] for r in rows]
    matrix = np.vstack([
        np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
    ])
    return ids, matrix


def cluster_repos(db_path: Path = DB_PATH, n_clusters: int | None = None,
                  random_state: int = 42) -> int:
    """
    KMeans-cluster all embedded repos and write cluster_id back to DB.
    Returns the number of clusters created.
    """
    ids, matrix = _load_embeddings(db_path)
    if len(ids) < 2:
        return 0

    # Normalise rows so cosine distance ≈ euclidean distance for KMeans
    matrix = normalize(matrix, norm="l2")

    if n_clusters is None:
        # Heuristic: one cluster per ~8 repos, min 2, max 50
        n_clusters = max(2, min(50, len(ids) // 8))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(matrix)

    for repo_id, label in zip(ids, labels):
        update_cluster(repo_id, int(label), db_path)

    return n_clusters


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = cluster_repos()
    print(f"Assigned {n} clusters")
