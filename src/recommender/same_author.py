from __future__ import annotations
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class SameAuthor(RecommendationStrategy):
    name = "same_author"
    label = "More by This Author"
    description = "Other books by the same author"

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        stats = self.loader.book_stats
        if stats.empty:
            return []

        # Look up the seed book's author
        seed_row = stats[stats["ISBN"] == seed_isbn]
        if seed_row.empty:
            return []
        author = seed_row.iloc[0].get("Book-Author", "")
        if not author:
            return []

        # Filter to same author, exclude the seed, sort by bayesian_rating desc
        same = stats[(stats["Book-Author"] == author) & (stats["ISBN"] != seed_isbn)]
        if same.empty:
            return []

        same = same.sort_values("bayesian_rating", ascending=False)

        results: list[Recommendation] = []
        for _, row in same.head(top_k).iterrows():
            results.append(
                self._build_recommendation(row["ISBN"], float(row["bayesian_rating"]))
            )
        return results
