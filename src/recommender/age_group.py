from __future__ import annotations
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class AgeGroup(RecommendationStrategy):
    name = "age_group"
    label = "Popular With Similar Readers"
    description = "Top-rated books among the age group that most enjoys your seed book."

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        dominant = self.loader.age_group_dominant
        top_books = self.loader.age_group_top_books

        if dominant.empty or top_books.empty:
            return []

        # Find the dominant age group for the seed book
        seed_row = dominant[dominant["ISBN"] == seed_isbn]
        if seed_row.empty:
            return []
        age_group = seed_row.iloc[0]["dominant_age_group"]

        # Get top books in that age group, excluding the seed
        group_books = top_books[
            (top_books["age_group"] == age_group) & (top_books["ISBN"] != seed_isbn)
        ]
        if group_books.empty:
            return []

        group_books = group_books.sort_values("bayesian_rating", ascending=False)

        results: list[Recommendation] = []
        for _, row in group_books.head(top_k).iterrows():
            results.append(
                self._build_recommendation(row["ISBN"], float(row["bayesian_rating"]))
            )
        return results
