"""
Rescan existing AI/ML repos for provider detection.

Usage:
  python -m pipeline.rescan           # show stats, then run text-only pass (free, no API calls)
  python -m pipeline.rescan --stats   # show stats only
  python -m pipeline.rescan --reset   # clear ai_providers so next pipeline run re-fetches dep files
"""
import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from pipeline.detect import detect_from_text
from pipeline.store import DB_PATH, get_connection


def print_stats() -> None:
    with get_connection() as conn:
        total_ai = conn.execute(
            "SELECT COUNT(*) FROM repos WHERE domain = 'ai_ml'"
        ).fetchone()[0]

        not_scanned = conn.execute(
            "SELECT COUNT(*) FROM repos WHERE domain = 'ai_ml' AND ai_providers IS NULL"
        ).fetchone()[0]

        empty = conn.execute("""
            SELECT COUNT(*) FROM repos
            WHERE domain = 'ai_ml'
              AND ai_providers IS NOT NULL
              AND json_extract(ai_providers, '$.frontier')    = json('[]')
              AND json_extract(ai_providers, '$.open_weight') = json('[]')
              AND json_extract(ai_providers, '$.frameworks')  = json('[]')
        """).fetchone()[0]

        has_frontier = conn.execute("""
            SELECT COUNT(*) FROM repos
            WHERE domain = 'ai_ml'
              AND json_extract(ai_providers, '$.frontier') != json('[]')
        """).fetchone()[0]

        has_open_weight = conn.execute("""
            SELECT COUNT(*) FROM repos
            WHERE domain = 'ai_ml'
              AND json_extract(ai_providers, '$.open_weight') != json('[]')
        """).fetchone()[0]

        has_framework_only = conn.execute("""
            SELECT COUNT(*) FROM repos
            WHERE domain = 'ai_ml'
              AND json_extract(ai_providers, '$.frontier')    = json('[]')
              AND json_extract(ai_providers, '$.open_weight') = json('[]')
              AND json_extract(ai_providers, '$.frameworks')  != json('[]')
        """).fetchone()[0]

    print(f"AI/ML repos total:             {total_ai}")
    print(f"  Not yet scanned:             {not_scanned}")
    print(f"  Scanned — nothing found:     {empty}")
    print(f"  Scanned — frontier found:    {has_frontier}")
    print(f"  Scanned — open-weight found: {has_open_weight}")
    print(f"  Scanned — framework only:    {has_framework_only}")
    scanned = total_ai - not_scanned
    detected = has_frontier + has_open_weight
    if scanned:
        print(f"\n  Detection rate: {detected}/{scanned} scanned repos ({detected/scanned:.0%})")


def rescan_text_only() -> None:
    """
    Merge README text signals into all ai_ml repos that have a stored README.
    Makes zero network calls — works entirely on data already in the DB.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, readme_text, ai_providers
            FROM repos
            WHERE domain = 'ai_ml'
              AND readme_text IS NOT NULL
        """).fetchall()
    rows = [dict(r) for r in rows]

    updated = 0
    for row in rows:
        text_found = detect_from_text(row["readme_text"])
        if not any(text_found.get(t) for t in ("frontier", "open_weight", "frameworks")):
            continue

        existing = (json.loads(row["ai_providers"])
                    if row["ai_providers"]
                    else {"frontier": [], "open_weight": [], "frameworks": []})
        merged = {
            tier: sorted(set(existing.get(tier, [])) | set(text_found.get(tier, [])))
            for tier in ("frontier", "open_weight", "frameworks")
        }

        if merged != existing:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE repos SET ai_providers = ? WHERE id = ?",
                    (json.dumps(merged), row["id"])
                )
            print(f"  {row['id']}: {merged}")
            updated += 1

    print(f"\nText-scan pass complete — {updated} of {len(rows)} README-scanned repos updated.")


def reset_for_full_rescan() -> None:
    """
    Clear ai_providers for all ai_ml repos.
    The next pipeline run will re-fetch dep files with the expanded package lists.
    """
    with get_connection() as conn:
        n = conn.execute(
            "UPDATE repos SET ai_providers = NULL WHERE domain = 'ai_ml'"
        ).rowcount
    print(f"Cleared ai_providers for {n} ai_ml repos.")
    print("Run the pipeline (`python -m pipeline.run`) to re-scan dep files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescan AI/ML repos for provider detection.")
    parser.add_argument("--stats",  action="store_true", help="Show breakdown and exit")
    parser.add_argument("--reset",  action="store_true",
                        help="Clear ai_providers so next pipeline run re-fetches dep files")
    args = parser.parse_args()

    print_stats()
    print()

    if args.stats:
        sys.exit(0)

    if args.reset:
        reset_for_full_rescan()
    else:
        print("Running text-only pass on stored READMEs...")
        rescan_text_only()
