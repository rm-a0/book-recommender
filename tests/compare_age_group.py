"""compare_age_group.py — AgeGroup strategy vs. pandas join baseline.

What this shows:
  The AgeGroup strategy looks up the seed's dominant age bracket in the
  age_group_dominant table, then pulls top-rated books for that bracket
  from age_group_top_books. The query baseline reproduces this as a direct
  pandas join. Results should be IDENTICAL — a correctness check.

Run:
    python tests/compare_age_group.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.age_group import AgeGroup

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "harry potter and the sorcerer's stone"
TOP_K = 10


def ml_recommendations(strategy: AgeGroup, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """Direct pandas equivalent: join dominant table + top_books table."""
    dominant = loader.age_group_dominant
    top_books = loader.age_group_top_books
    stats = loader.book_stats

    # Step 1: find seed's dominant age group
    dom_row = dominant[dominant["ISBN"] == seed_isbn]
    if dom_row.empty:
        print(f"[WARN] Seed ISBN {seed_isbn!r} not found in age_group_dominant")
        return []
    group = dom_row.iloc[0]["dominant_age_group"]
    print(f"  [Query] Seed's dominant age group: {group!r}")

    # Step 2: top books in that group, exclude seed
    group_books = (
        top_books[(top_books["age_group"] == group) & (top_books["ISBN"] != seed_isbn)]
        .sort_values("bayesian_rating", ascending=False)
        .head(TOP_K)
    )

    # Step 3: join with book_stats to get title/author
    merged = group_books.merge(
        stats[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left",
    )

    return [
        {
            "isbn": r["ISBN"],
            "title": r.get("Book-Title", ""),
            "author": r.get("Book-Author", ""),
            "score": float(r["bayesian_rating"]),
        }
        for _, r in merged.iterrows()
    ]


def main() -> None:
    print("Loading artifacts...")
    loader = load_artifacts()
    service = RecommenderService(loader=loader)

    result = find_seed(service, SEED_TITLE)
    if result is None:
        sys.exit(1)
    isbn, title, author = result

    print_header("AgeGroup — Popular With Similar Readers", title, author, isbn)
    print(
        "\nApproach: Find the seed book's dominant age bracket from age_group_dominant, "
        "then return top bayesian_rating books in that bracket from age_group_top_books.\n"
        "The query baseline does the same with a direct pandas join.\n"
        "Expectation: results should be IDENTICAL.\n"
    )

    strategy = AgeGroup(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="AgeGroup strategy",
        query_label="pandas join (dominant → top_books)",
        score_col="score",
    )


if __name__ == "__main__":
    main()
