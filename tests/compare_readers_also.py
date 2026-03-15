"""compare_readers_also.py — ReadersAlso (CF) vs. raw co-rating query baseline.

What this shows:
  ReadersAlso does an O(1) sparse matrix row lookup into a precomputed
  item-item adjusted cosine similarity matrix. The CF matrix was built by:
    1. Mean-centering ratings per user (removing user-bias)
    2. Computing dot products over all co-rated book pairs
    3. Dividing by L2 norms → adjusted cosine similarity

  The query baseline replicates the UNCENTERED version from raw ratings:
  for each book co-rated with the seed by at least one user, count how many
  users co-rated both and sum their ratings as a naive overlap score. This
  intentionally does NOT mean-center, so it captures the raw co-occurrence
  signal without bias removal.

  Differences between ML and query results reveal where mean-centering
  changes the ranking: books rated identically by all users (generic
  crowd-pleasers) rank lower in the CF matrix than the naive count suggests.

Run:
    python tests/compare_readers_also.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.readers_also import ReadersAlso
from src.config import PROCESSED_DATA_PATH

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "the fellowship of the ring"
TOP_K = 10


def ml_recommendations(strategy: ReadersAlso, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """
    Naive co-rating baseline (no mean-centering):
    1. Find all users who rated the seed book
    2. Find all other books those users also rated
    3. Score = number of common raters (simple co-occurrence count)
    4. Sort descending, take TOP_K
    """
    ratings_path = os.path.join(PROCESSED_DATA_PATH, "ratings.parquet")
    if not os.path.exists(ratings_path):
        print(f"[ERROR] ratings.parquet not found at {ratings_path}")
        return []

    ratings = pd.read_parquet(ratings_path, columns=["User-ID", "ISBN", "Book-Rating"])

    # Users who rated the seed
    seed_users = set(ratings[ratings["ISBN"] == seed_isbn]["User-ID"])
    print(f"  [Query] Users who rated seed: {len(seed_users)}")

    if not seed_users:
        print(f"  [Query] No users found for seed ISBN {seed_isbn!r}")
        return []

    # Other books those users rated
    co_rated = ratings[
        ratings["User-ID"].isin(seed_users) & (ratings["ISBN"] != seed_isbn)
    ]

    # Score: count co-raters + mean rating (break ties)
    agg = (
        co_rated.groupby("ISBN")
        .agg(co_rater_count=("User-ID", "count"), mean_rating=("Book-Rating", "mean"))
        .reset_index()
        .sort_values(["co_rater_count", "mean_rating"], ascending=[False, False])
        .head(TOP_K)
    )

    # Join with book_stats for title/author
    stats = loader.book_stats[["ISBN", "Book-Title", "Book-Author"]]
    merged = agg.merge(stats, on="ISBN", how="left")

    return [
        {
            "isbn": r["ISBN"],
            "title": r.get("Book-Title", ""),
            "author": r.get("Book-Author", ""),
            "score": float(r["co_rater_count"]),
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

    print_header("ReadersAlso — Readers Also Enjoyed", title, author, isbn)
    print(
        "\nApproach comparison:\n"
        "  ML   : O(1) lookup into precomputed adjusted cosine CF matrix.\n"
        "         Ratings are mean-centered per user before computing dot products,\n"
        "         so the similarity reflects shared TASTE not just co-occurrence.\n"
        "  Query: Raw co-occurrence count from ratings.parquet.\n"
        "         Simply counts how many users rated both books (no centering).\n"
        "\nExpected differences: popular books that many users rate (regardless of\n"
        "whether they liked them) rank higher in the query baseline. The CF matrix\n"
        "downweights these in favor of books users specifically liked together.\n"
    )

    strategy = ReadersAlso(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="ReadersAlso CF (adjusted cosine similarity)",
        query_label="Co-occurrence count (raw, no mean-centering)",
        score_col="score",
    )


if __name__ == "__main__":
    main()
