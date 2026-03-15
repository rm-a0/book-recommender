"""compare_similar_themes.py — SimilarThemes (FAISS semantic) vs. Jaccard subject overlap.

What this shows:
  SimilarThemes encodes each book as a 384-dim MiniLM sentence embedding of
  "Title by Author. Description. Genres: subjects" and queries FAISS for the
  nearest neighbours by cosine similarity. This captures semantic meaning —
  books can match even if they share no keywords.

  The query baseline computes Jaccard similarity on the raw subject-tag sets
  from book_metadata_enriched.parquet. This is a keyword overlap approach:
  books only match if they share the same subject strings exactly.

  Differences reveal where semantic embeddings generalise beyond exact keyword
  matching: e.g. "epic fantasy quest" and "heroic journey" share no words but
  similar embeddings.

Run:
    python tests/compare_similar_themes.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.similar_themes import SimilarThemes

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "the fellowship of the ring"
TOP_K = 10
MIN_SUBJECTS_FOR_QUERY = 1  # skip books with no subjects in the query baseline


def ml_recommendations(strategy: SimilarThemes, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def _subject_set(subjects_str: str | None) -> set[str]:
    if not subjects_str or not isinstance(subjects_str, str):
        return set()
    return {s.strip().lower() for s in subjects_str.split(",") if s.strip()}


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """
    Jaccard similarity on subject tags from book_metadata_enriched.parquet.
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    meta = loader.enriched_metadata
    if meta.empty:
        print("[WARN] enriched_metadata not loaded")
        return []

    seed_row = meta[meta["ISBN"] == seed_isbn]
    if seed_row.empty:
        print(f"[WARN] Seed ISBN {seed_isbn!r} not in enriched metadata")
        return []

    seed_subjects = _subject_set(seed_row.iloc[0].get("subjects"))
    print(f"  [Query] Seed subjects ({len(seed_subjects)}): {', '.join(list(seed_subjects)[:8])}")

    if not seed_subjects:
        print("  [Query] Seed has no subjects — Jaccard query will return nothing.")
        return []

    results = []
    for _, row in meta.iterrows():
        if row["ISBN"] == seed_isbn:
            continue
        other_subjects = _subject_set(row.get("subjects"))
        if len(other_subjects) < MIN_SUBJECTS_FOR_QUERY:
            continue
        intersection = seed_subjects & other_subjects
        if not intersection:
            continue
        union = seed_subjects | other_subjects
        jaccard = len(intersection) / len(union)
        results.append({
            "isbn": row["ISBN"],
            "title": row.get("Book-Title", ""),
            "author": row.get("Book-Author", ""),
            "score": jaccard,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]


def main() -> None:
    print("Loading artifacts...")
    loader = load_artifacts()
    service = RecommenderService(loader=loader)

    result = find_seed(service, SEED_TITLE)
    if result is None:
        sys.exit(1)
    isbn, title, author = result

    print_header("SimilarThemes — Similar Themes and Style", title, author, isbn)
    print(
        "\nApproach comparison:\n"
        "  ML   : FAISS nearest-neighbour search over MiniLM sentence embeddings.\n"
        "         Embedding text: 'Title by Author. Description. Genres: subjects'\n"
        "         Score is cosine similarity (L2-normalised, so inner product = cosine).\n"
        "  Query: Jaccard similarity on subject tag sets from enriched metadata.\n"
        "         Only exact string matches in subjects count as overlap.\n"
        "\nExpected: FAISS finds semantically related books even without shared subjects.\n"
        "Jaccard requires exact tag overlap — good recall for genre but misses\n"
        "thematic similarity expressed differently (e.g. 'dragons' vs 'mythical beasts').\n"
    )

    strategy = SimilarThemes(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="SimilarThemes FAISS (cosine similarity on MiniLM embeddings)",
        query_label="Jaccard similarity on subject tags",
        score_col="score",
    )


if __name__ == "__main__":
    main()
