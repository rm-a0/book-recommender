from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .loader import ArtifactLoader

@dataclass
class Recommendation:
    """A single recommended book returned by a strategy."""
    isbn: str
    title: str
    author: str
    score: float
    strategy: str
    # Optional fields populated by the loader
    description: str = ""
    subjects: str = ""
    rating_count: int = 0
    bayesian_rating: float = 0.0

class RecommendationStrategy(ABC):
    """Interface that every recommendation strategy must implement."""
    name: str          # internal (not shown to users)
    label: str         # external (human-readable, shown in UI)
    description: str

    def __init__(self, loader: ArtifactLoader) -> None:
        self.loader = loader

    @abstractmethod
    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        """Return up to *top_k* recommendations for the given seed book."""
        ...

    def _isbn_to_metadata(self, isbn: str) -> dict:
        """Look up title/author/stats for an ISBN from book_stats."""
        stats = self.loader.book_stats
        row = stats[stats["ISBN"] == isbn]
        if row.empty:
            return {
                "title": "", 
                "author": "", 
                "rating_count": 0, 
                "bayesian_rating": 0.0
            }
        r = row.iloc[0]
        return {
            "title": r.get("Book-Title", ""),
            "author": r.get("Book-Author", ""),
            "rating_count": int(r.get("rating_count", 0)),
            "bayesian_rating": float(r.get("bayesian_rating", 0.0)),
        }

    def _build_recommendation(self, isbn: str, score: float) -> Recommendation:
        """Create a Recommendation object with metadata populated from artifacts."""
        meta = self._isbn_to_metadata(isbn)
        return Recommendation(
            isbn=isbn,
            title=meta["title"],
            author=meta["author"],
            score=round(score, 6),
            strategy=self.name,
            rating_count=meta["rating_count"],
            bayesian_rating=meta["bayesian_rating"],
        )
