"""compare_hidden_gems.py — HiddenGems strategy vs. popularity-penalised query baseline.

What this shows:
  HiddenGems takes the top-100 CF candidates for the seed book and re-ranks
  them by: gem_score = cf_similarity / log(1 + rating_count). Books that have
  high CF similarity but few ratings (low exposure) bubble to the top.

  The query baseline starts from raw co-occurrence counts (same as
  compare_readers_also.py) and applies the same log-penalty formula. This
  directly shows what the penalty does: it demotes crowd-pleasers that
  happen to be co-read a lot and promotes niche titles with concentrated
  readership overlap.

  Differences between ML and query results reveal where mean-centering
  in the CF matrix (which the query baseline lacks) interacts with the
  popularity penalty.

Run:
    python tests/compare_hidden_gems.py
"""

from __future__ import annotations

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from tests._helpers import load_artifacts, find_seed, print_header, print_comparison
from src.recommender.service import RecommenderService
from src.recommender.hidden_gems import HiddenGems
from src.config import PROCESSED_DATA_PATH, HIDDEN_GEM_CANDIDATE_POOL

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_TITLE = "the fellowship of the ring"
TOP_K = 10


def ml_recommendations(strategy: HiddenGems, isbn: str) -> list[dict]:
    recs = strategy.recommend(isbn, top_k=TOP_K)
    return [
        {"isbn": r.isbn, "title": r.title, "author": r.author, "score": r.score}
        for r in recs
    ]


def query_recommendations(loader, seed_isbn: str) -> list[dict]:
    """
    Popularity-penalised co-occurrence baseline:
    1. Find co-raters of the seed (same as readers_also baseline)
    2. Compute raw co-rater count as a proxy for CF similarity
    3. Apply gem_score = co_rater_count / log(1 + rating_count)
    4. Sort by gem_score descending, take TOP_K
    """
    ratings_path = os.path.join(PROCESSED_DATA_PATH, "ratings.parquet")
    if not os.path.exists(ratings_path):
        print(f"[ERROR] ratings.parquet not found at {ratings_path}")
        return []

    ratings = pd.read_parquet(ratings_path, columns=["User-ID", "ISBN", "Book-Rating"])
    stats = loader.book_stats[["ISBN", "Book-Title", "Book-Author", "rating_count"]]

    seed_users = set(ratings[ratings["ISBN"] == seed_isbn]["User-ID"])
    print(f"  [Query] Users who rated seed: {len(seed_users)}")

    if not seed_users:
        return []

    co_rated = ratings[
        ratings["User-ID"].isin(seed_users) & (ratings["ISBN"] != seed_isbn)
    ]

    agg = (
        co_rated.groupby("ISBN")
        .agg(co_rater_count=("User-ID", "count"))
        .reset_index()
        .sort_values("co_rater_count", ascending=False)
        .head(HIDDEN_GEM_CANDIDATE_POOL)
    )

    # Join rating_count from book_stats
    merged = agg.merge(stats, on="ISBN", how="left")

    # Apply penalty: gem_score = co_rater_count / log(1 + rating_count)
    merged["gem_score"] = merged["co_rater_count"] / merged["rating_count"].apply(
        lambda rc: math.log(1 + float(rc)) if pd.notna(rc) and rc > 0 else 1.0
    )

    top = merged.sort_values("gem_score", ascending=False).head(TOP_K)

    return [
        {
            "isbn": r["ISBN"],
            "title": r.get("Book-Title", ""),
            "author": r.get("Book-Author", ""),
            "score": float(r["gem_score"]),
        }
        for _, r in top.iterrows()
    ]


def main() -> None:
    print("Loading artifacts...")
    loader = load_artifacts()
    service = RecommenderService(loader=loader)

    result = find_seed(service, SEED_TITLE)
    if result is None:
        sys.exit(1)
    isbn, title, author = result

    print_header("HiddenGems — Hidden Gems", title, author, isbn)
    print(
        "\nApproach comparison:\n"
        "  ML   : Top-100 CF candidates re-ranked by cf_sim / log(1 + rating_count).\n"
        "         CF similarity is adjusted cosine (mean-centered ratings).\n"
        "  Query: Top co-occurrence candidates re-ranked by the same penalty formula.\n"
        "         Similarity proxy is raw co-rater count (not mean-centered).\n"
        "\nExpected: both surfaces niche books. Differences come from mean-centering\n"
        "— the CF matrix may rate a book higher if users consistently gave it above\n"
        "their average, even if fewer users read it alongside the seed.\n"
    )

    strategy = HiddenGems(loader)
    ml_rows = ml_recommendations(strategy, isbn)
    query_rows = query_recommendations(loader, isbn)

    print_comparison(
        ml_rows,
        query_rows,
        ml_label="HiddenGems strategy (CF sim / log(1 + rating_count))",
        query_label="Penalised co-occurrence (co_raters / log(1 + rating_count))",
        score_col="score",
    )


if __name__ == "__main__":
    main()
