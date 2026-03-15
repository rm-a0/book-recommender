"""Shared helpers for strategy comparison scripts.

Each comparison script follows the same pattern:
1. Load ML artifacts via ArtifactLoader
2. Load raw / processed parquets via pandas
3. Run the strategy (ML approach)
4. Run an equivalent pandas/numpy query (baseline)
5. Print a side-by-side comparison

Run any script directly, e.g.:
    python tests/compare_same_author.py
"""
from __future__ import annotations
import os
import sys

# Allow running scripts from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.recommender.loader import ArtifactLoader
from src.recommender.service import RecommenderService


def load_artifacts() -> ArtifactLoader:
    loader = ArtifactLoader()
    loader.load()
    return loader


def find_seed(service: RecommenderService, title: str) -> tuple[str, str, str] | None:
    """Search for a book by title and return (isbn, title, author) of the top match."""
    matches = service.search_books(title, max_results=5)
    if not matches:
        print(f"[ERROR] No book found matching '{title}'")
        return None
    top = matches[0]
    return top.isbn, top.title, top.author


def print_header(strategy_name: str, seed_title: str, seed_author: str, seed_isbn: str) -> None:
    width = 80
    print("=" * width)
    print(f"  Strategy : {strategy_name}")
    print(f"  Seed     : {seed_title!r}  by {seed_author}")
    print(f"  ISBN     : {seed_isbn}")
    print("=" * width)


def print_comparison(
    ml_rows: list[dict],
    query_rows: list[dict],
    ml_label: str = "ML strategy result",
    query_label: str = "Query baseline",
    score_col: str = "score",
) -> None:
    """Print ML results and query results side by side, then note differences."""

    col_w = 38

    def fmt_row(rank: int, title: str, author: str, score: float) -> str:
        entry = f"{rank:>2}. {title[:30]!s:<30} | {author[:20]!s}"
        return f"{entry:<{col_w}}  {score:.4f}"

    header = f"{'Rank.  Title                           | Author':<{col_w}}  Score"
    divider = "-" * (col_w * 2 + 10)

    print(f"\n{'--- ' + ml_label + ' ---':^80}")
    print(header)
    print(divider)
    for i, r in enumerate(ml_rows, 1):
        print(fmt_row(i, r.get("title", ""), r.get("author", ""), r.get(score_col, 0.0)))

    print(f"\n{'--- ' + query_label + ' ---':^80}")
    print(header)
    print(divider)
    for i, r in enumerate(query_rows, 1):
        print(fmt_row(i, r.get("title", ""), r.get("author", ""), r.get(score_col, 0.0)))

    # Overlap analysis
    ml_isbns = [r["isbn"] for r in ml_rows]
    q_isbns = [r["isbn"] for r in query_rows]
    overlap = set(ml_isbns) & set(q_isbns)
    only_ml = set(ml_isbns) - set(q_isbns)
    only_q = set(q_isbns) - set(ml_isbns)

    print(f"\n--- Overlap Analysis ---")
    print(f"  Books in both lists      : {len(overlap)}")
    print(f"  Only in ML result        : {len(only_ml)}")
    print(f"  Only in query baseline   : {len(only_q)}")
    if only_ml:
        titles = [r["title"] for r in ml_rows if r["isbn"] in only_ml]
        print(f"  ML-only titles           : {', '.join(t[:25] for t in titles[:5])}")
    if only_q:
        titles = [r["title"] for r in query_rows if r["isbn"] in only_q]
        print(f"  Query-only titles        : {', '.join(t[:25] for t in titles[:5])}")
    print()
