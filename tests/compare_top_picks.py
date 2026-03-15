"""compare_top_picks.py — TopPicks (CF + semantic fusion) vs. bayesian rating baseline.

What this shows:
  TopPicks fuses CF and semantic signals with a weighted linear combination
  (alpha=0.6 CF / 0.4 semantic), adds a Reciprocal Rank Fusion (RRF) bonus
  for books appearing in both lists, then applies a mild log popularity penalty.

  The query baseline takes a simple union of:
    - Same-author books sorted by bayesian_rating
    - Books with overlapping subjects sorted by bayesian_rating
  scored by their bayesian_rating. This represents what a straightforward
  "best books that match this one" query would return without any ML.

  Differences show where multi-signal fusion surfaces books that neither
  individual signal would rank highly — especially books a niche of users
  loved (high CF sim) AND that match thematically (high semantic sim),
  but are not globally top-rated.

Run:
    python tests/compare_top_picks.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.top_picks import TopPicks

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "the fellowship of the ring"
TOP_K = 10
MAX_SUBJECT_CANDIDATES = 200  # cap candidate pool for query baseline performance


def ml_recommendations(strategy: TopPicks, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def _subject_set(subjects_str) -> set[str]:
    if not subjects_str or not isinstance(subjects_str, str):
        return set()
    return {s.strip().lower() for s in subjects_str.split(",") if s.strip()}


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """
    Simple quality-based query baseline (no ML):
    1. Collect same-author books (sorted by bayesian_rating)
    2. Collect books with any overlapping subject tags (sorted by bayesian_rating)
    3. Union the two sets; score = bayesian_rating; deduplicate; sort; take TOP_K
    """
    stats = loader.book_stats
    meta = loader.enriched_metadata

    # Resolve seed metadata
    seed_stats = stats[stats["ISBN"] == seed_isbn]
    if seed_stats.empty:
        print(f"[WARN] Seed ISBN {seed_isbn!r} not in book_stats")
        return []
    seed_author = seed_stats.iloc[0]["Book-Author"]

    # Part 1: same-author books
    same_author = (
        stats[(stats["Book-Author"] == seed_author) & (stats["ISBN"] != seed_isbn)]
        [["ISBN", "Book-Title", "Book-Author", "bayesian_rating"]]
    )

    # Part 2: subject-overlap books
    subject_candidates = pd.DataFrame()
    if not meta.empty:
        seed_meta = meta[meta["ISBN"] == seed_isbn]
        if not seed_meta.empty:
            seed_subjects = _subject_set(seed_meta.iloc[0].get("subjects"))
            if seed_subjects:
                def has_overlap(subjects_str):
                    return bool(seed_subjects & _subject_set(subjects_str))

                subject_mask = meta["subjects"].apply(has_overlap)
                subject_isbns = set(meta[subject_mask]["ISBN"]) - {seed_isbn}
                print(f"  [Query] Books with subject overlap: {len(subject_isbns)}")

                # Limit pool size for performance
                subject_candidates = (
                    stats[stats["ISBN"].isin(subject_isbns)]
                    [["ISBN", "Book-Title", "Book-Author", "bayesian_rating"]]
                    .sort_values("bayesian_rating", ascending=False)
                    .head(MAX_SUBJECT_CANDIDATES)
                )

    # Union, deduplicate, sort
    combined = (
        pd.concat([same_author, subject_candidates], ignore_index=True)
        .drop_duplicates(subset="ISBN")
        .sort_values("bayesian_rating", ascending=False)
        .head(TOP_K)
    )

    return [
        {
            "isbn": r["ISBN"],
            "title": r.get("Book-Title", ""),
            "author": r.get("Book-Author", ""),
            "score": float(r["bayesian_rating"]),
        }
        for _, r in combined.iterrows()
    ]


def main() -> None:
    print("Loading artifacts...")
    loader = load_artifacts()
    service = RecommenderService(loader=loader)

    result = find_seed(service, SEED_TITLE)
    if result is None:
        sys.exit(1)
    isbn, title, author = result

    print_header("TopPicks — Top Picks For You", title, author, isbn)
    print(
        "\nApproach comparison:\n"
        "  ML   : Linear fusion of CF (weight=0.6) and semantic (weight=0.4) scores,\n"
        "         both normalised to [0,1]. RRF bonus for overlap between lists.\n"
        "         Mild log popularity penalty: fused / log(1 + 0.3 * rating_count).\n"
        "  Query: Union of same-author + subject-overlap books, scored by bayesian_rating.\n"
        "         No CF signal, no semantic signal — pure quality + attribute matching.\n"
        "\nExpected: TopPicks surfaces books that users who liked this one also liked,\n"
        "regardless of author or genre tag. Query baseline finds structurally similar\n"
        "books but misses community-validated taste signals from co-ratings.\n"
    )

    strategy = TopPicks(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="TopPicks (CF + semantic fusion + RRF + popularity penalty)",
        query_label="Bayesian rating of same-author + subject-overlap union",
        score_col="score",
    )


if __name__ == "__main__":
    main()
