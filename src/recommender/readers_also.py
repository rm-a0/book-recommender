from __future__ import annotations
import numpy as np
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class ReadersAlso(RecommendationStrategy):
    name = "readers_also"
    label = "Readers Also Enjoyed"
    description = "Books that users who read this book also enjoyed"

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        if not self.loader.has_cf:
            return []

        idx = self.loader.isbn_index.get(seed_isbn)
        if idx is None:
            return []

        # Get similarity scores for all other books from the sparse matrix
        row = self.loader.item_similarity.getrow(idx).toarray().flatten()
        if row.max() == 0:
            return []

        # Pull a wider candidate pool so skipping self/invalid entries still returns top_k items.
        pool_size = min(len(row), max(top_k * 10, 50))
        top_indices = np.argsort(row)[::-1][:pool_size]
        results: list[Recommendation] = []
        for col_idx in top_indices:
            sim = float(row[col_idx])
            if sim <= 0:
                break
            isbn = self.loader.index_isbn.get(int(col_idx))
            if isbn is None or isbn == seed_isbn:
                continue
            results.append(self._build_recommendation(isbn, sim))
            if len(results) >= top_k:
                break

        return results[:top_k]
