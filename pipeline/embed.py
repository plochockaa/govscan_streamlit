import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from pipeline.store import DB_PATH, get_connection, get_unembedded, update_embedding

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def build_text_for_embedding(repo: dict) -> str:
    topics_raw = repo.get("topics") or "[]"
    if isinstance(topics_raw, list):
        topics = ", ".join(topics_raw)
    else:
        topics = ", ".join(json.loads(topics_raw))

    parts = [
        repo.get("name") or "",
        repo.get("description") or "",
        repo.get("llm_summary") or "",
        topics,
    ]
    return " | ".join(p for p in parts if p)


def embed_and_store(db_path: Path = DB_PATH) -> int:
    """Embed every classified-but-unembedded repo. Returns number embedded."""
    repos = get_unembedded(db_path)
    if not repos:
        return 0

    texts = [build_text_for_embedding(r) for r in repos]
    vecs = list(_get_model().embed(texts))

    for repo, vec in zip(repos, vecs):
        update_embedding(repo["id"], np.array(vec, dtype=np.float32).tobytes(), db_path)

    return len(repos)


def find_similar(query_id: str, top_k: int = 5,
                 db_path: Path = DB_PATH) -> list[dict]:
    """Return top-k repos most similar to query_id by cosine similarity."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT embedding FROM repos WHERE id = ?", (query_id,)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return []
        query_vec = np.frombuffer(row["embedding"], dtype=np.float32)

        rows = conn.execute("""
            SELECT id, name, org, country, domain, llm_summary, stars, embedding
            FROM repos
            WHERE embedding IS NOT NULL AND id != ?
        """, (query_id,)).fetchall()

    if not rows:
        return []

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    candidates = []
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        norm = vec / (np.linalg.norm(vec) + 1e-9)
        score = float(np.dot(query_norm, norm))
        entry = {k: r[k] for k in r.keys() if k != "embedding"}
        entry["similarity"] = round(score, 4)
        candidates.append(entry)

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]
