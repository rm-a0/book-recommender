"""compare_same_author.py — SameAuthor strategy vs. pandas query baseline.

What this shows:
  The SameAuthor strategy does an exact match on Book-Author and sorts by
  bayesian_rating. The query baseline replicates the same logic directly in
  pandas. These two should produce **identical** results — this script
  validates correctness of the strategy implementation.

Run:
    python tests/compare_same_author.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.same_author import SameAuthor

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "the fellowship of the ring"
TOP_K = 10


def ml_recommendations(strategy: SameAuthor, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """Direct pandas equivalent: filter book_stats by author, sort bayesian_rating."""
    stats = loader.book_stats

    # Resolve seed author
    row = stats[stats["ISBN"] == seed_isbn]
    if row.empty:
        print("[WARN] Seed ISBN not found in book_stats")
        return []
    seed_author = row.iloc[0]["Book-Author"]

    same = (
        stats[stats["Book-Author"] == seed_author]
        .loc[stats["ISBN"] != seed_isbn]
        .sort_values("bayesian_rating", ascending=False)
        .head(TOP_K)
    )

    return [
        {
            "isbn": r["ISBN"],
            "title": r["Book-Title"],
            "author": r["Book-Author"],
            "score": float(r["bayesian_rating"]),
        }
        for _, r in same.iterrows()
    ]


def main() -> None:
    print("Loading artifacts...")
    loader = load_artifacts()
    service = RecommenderService(loader=loader)

    result = find_seed(service, SEED_TITLE)
    if result is None:
        sys.exit(1)
    isbn, title, author = result

    print_header("SameAuthor — More by This Author", title, author, isbn)
    print(
        "\nApproach: The ML strategy does an exact match on Book-Author (lowercased during "
        "ingestion) and sorts by precomputed bayesian_rating.\n"
        "The query baseline reproduces the same logic in pandas.\n"
        "Expectation: results should be IDENTICAL — any difference indicates a bug.\n"
    )

    strategy = SameAuthor(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="SameAuthor strategy",
        query_label="pandas query (filter by author, sort bayesian_rating)",
        score_col="score",
    )


if __name__ == "__main__":
    main()
