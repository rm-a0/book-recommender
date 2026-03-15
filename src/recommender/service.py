from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from .base import Recommendation
from .loader import ArtifactLoader
from .registry import StrategyRegistry, create_default_registry

@dataclass
class BookMatch:
    """A book returned by a search query."""
    isbn: str
    title: str
    author: str
    rating_count: int
    bayesian_rating: float

@dataclass
class RecommendationResult:
    """Recommendations from a single strategy, with strategy metadata attached."""
    strategy_name: str
    strategy_label: str
    strategy_description: str
    recommendations: list[Recommendation] = field(default_factory=list)

class RecommenderService:
    """Facade over ArtifactLoader + StrategyRegistry.

    Construct once (at app startup) and reuse across requests - artifact
    loading is expensive and idempotent.
    """
    def __init__(
        self,
        loader: ArtifactLoader | None = None,
        registry: StrategyRegistry | None = None,
    ) -> None:
        self._loader = loader or ArtifactLoader().load()
        self._registry = registry or create_default_registry(self._loader)

    def search_books(self, query: str, max_results: int = 10) -> list[BookMatch]:
        """Return books whose title contains *query* (case-insensitive)."""
        stats = self._loader.book_stats
        if stats.empty:
            return []

        q = query.strip().lower()
        if not q:
            return []

        titles = stats["Book-Title"].fillna("").astype(str).str.lower()

        # Prioritize exact matches, then fuzzy matches, both sorted by rating_count
        exact_mask = titles == q
        exact = stats[exact_mask]
        fuzzy = stats[~exact_mask & titles.str.contains(q, regex=False)]

        combined = (
            pd.concat(
                [
                    exact.sort_values("rating_count", ascending=False),
                    fuzzy.sort_values("rating_count", ascending=False),
                ]
            )
            .head(max_results)
        )

        return [
            BookMatch(
                isbn=str(row["ISBN"]),
                title=str(row.get("Book-Title", "")),
                author=str(row.get("Book-Author", "")),
                rating_count=int(row.get("rating_count", 0)),
                bayesian_rating=float(row.get("bayesian_rating", 0.0)),
            )
            for _, row in combined.iterrows()
        ]

    def get_book(self, isbn: str) -> BookMatch | None:
        """Look up a single book by ISBN. Returns None if not found."""
        stats = self._loader.book_stats
        if stats.empty:
            return None
        row_df = stats[stats["ISBN"] == isbn]
        if row_df.empty:
            return None
        row = row_df.iloc[0]
        return BookMatch(
            isbn=isbn,
            title=str(row.get("Book-Title", "")),
            author=str(row.get("Book-Author", "")),
            rating_count=int(row.get("rating_count", 0)),
            bayesian_rating=float(row.get("bayesian_rating", 0.0)),
        )

    def recommend(
        self,
        isbn: str,
        strategy: str,
        top_k: int = 10,
    ) -> RecommendationResult:
        """Run a single strategy and return a typed result object."""
        strat_obj = self._registry.get(strategy)
        recs = strat_obj.recommend(isbn, top_k)
        return RecommendationResult(
            strategy_name=strat_obj.name,
            strategy_label=strat_obj.label,
            strategy_description=strat_obj.description,
            recommendations=recs,
        )

    def recommend_all(
        self,
        isbn: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Run every registered strategy and return results in registration order."""
        return [
            self.recommend(isbn, name, top_k)
            for name in self._registry._strategies
        ]

    def list_strategies(self) -> list[dict[str, str]]:
        """Return name/label/description for every registered strategy."""
        return self._registry.list_strategies()