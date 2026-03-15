from __future__ import annotations
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

        # Find the seed book's embedding index
        try:
            seed_idx = self.loader.embedding_isbn_map.index(seed_isbn)
        except ValueError:
            return []

        seed_emb = self.loader.book_embeddings[seed_idx : seed_idx + 1]  # (1, dim)

        # Query FAISS (request top_k+1 because the seed itself will appear)
        distances, indices = self.loader.faiss_index.search(seed_emb, top_k + 1)

        results: list[Recommendation] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.loader.embedding_isbn_map):
                continue
            isbn = self.loader.embedding_isbn_map[idx]
            if isbn == seed_isbn:
                continue
            score = float(dist)
            results.append(self._build_recommendation(isbn, score))

        return results[:top_k]
