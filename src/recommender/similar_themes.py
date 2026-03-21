from __future__ import annotations
from rapidfuzz import fuzz
from ..config import EDITION_SIMILARITY_THRESHOLD
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader


class SimilarThemes(RecommendationStrategy):
    name = "similar_themes"
    label = "Similar Themes and Style"
    description = "Books with similar descriptions and genres"

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        if not self.loader.has_embeddings:
            return []

        try:
            seed_idx = self.loader.embedding_isbn_map.index(seed_isbn)
        except ValueError:
            return []

        seed_emb = self.loader.book_embeddings[seed_idx : seed_idx + 1]

        # Request a larger pool to account for any edition-filtered results
        distances, indices = self.loader.faiss_index.search(seed_emb, top_k + 10)

        # Build title lookup for edition filtering
        stats = self.loader.book_stats
        title_map: dict[str, str] = {}
        if not stats.empty and "Book-Title" in stats.columns:
            title_map = dict(zip(stats["ISBN"], stats["Book-Title"].fillna("").str.lower()))

        seed_title = title_map.get(seed_isbn, "")

        results: list[Recommendation] = []
        for dist, idx in zip(distances[0], indices[0]):
            if len(results) >= top_k:
                break
            if idx < 0 or idx >= len(self.loader.embedding_isbn_map):
                continue
            isbn = self.loader.embedding_isbn_map[idx]
            if isbn == seed_isbn:
                continue
            # Skip books that are likely a different edition of the seed
            if seed_title:
                candidate_title = title_map.get(isbn, "")
                if candidate_title and fuzz.ratio(candidate_title, seed_title) >= EDITION_SIMILARITY_THRESHOLD:
                    continue
            results.append(self._build_recommendation(isbn, float(dist)))

        return results