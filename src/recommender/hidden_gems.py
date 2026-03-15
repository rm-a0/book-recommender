from __future__ import annotations
import math
import numpy as np
from ..config import HIDDEN_GEM_CANDIDATE_POOL
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class HiddenGems(RecommendationStrategy):
    name = "hidden_gems"
    label = "Hidden Gems"
    description = "Books that are similar to your seed but have far fewer ratings"

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        if not self.loader.has_cf:
            return []

        idx = self.loader.isbn_index.get(seed_isbn)
        if idx is None:
            return []

        # Get a larger CF candidate pool
        row = self.loader.item_similarity.getrow(idx).toarray().flatten()
        if row.max() == 0:
            return []

        pool_size = HIDDEN_GEM_CANDIDATE_POOL
        top_indices = np.argsort(row)[::-1][:pool_size]

        # Build rating_count lookup from book_stats
        stats = self.loader.book_stats
        count_map: dict[str, int] = {}
        if not stats.empty and "rating_count" in stats.columns:
            count_map = dict(zip(stats["ISBN"], stats["rating_count"]))

        # Rerank: hidden_gem_score = cf_similarity / log(1 + rating_count)
        scored: list[tuple[str, float, float]] = []
        for col_idx in top_indices:
            sim = float(row[col_idx])
            if sim <= 0:
                break
            isbn = self.loader.index_isbn.get(int(col_idx))
            if isbn is None or isbn == seed_isbn:
                continue
            rating_count = count_map.get(isbn, 1)
            gem_score = sim / math.log(1 + rating_count)
            scored.append((isbn, gem_score, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[Recommendation] = []
        for isbn, gem_score, _sim in scored[:top_k]:
            results.append(self._build_recommendation(isbn, gem_score))
        return results
