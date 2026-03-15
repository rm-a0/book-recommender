import math
import numpy as np
from ..config import FUSION_ALPHA, FUSION_CANDIDATE_POOL
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class TopPicks(RecommendationStrategy):
    name = "top_picks"
    label = "Top Picks For You"
    description = "Best recommendations combining collaborative and semantic signals."

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        cf_candidates = self._get_cf_candidates(seed_isbn)
        sem_candidates = self._get_semantic_candidates(seed_isbn)

        if not cf_candidates and not sem_candidates:
            return []

        # Normalize scores to [0, 1] within each candidate set
        cf_candidates = _normalize_scores(cf_candidates)
        sem_candidates = _normalize_scores(sem_candidates)

        # Build rank maps for reciprocal rank fusion
        cf_ranks = {isbn: rank for rank, (isbn, _) in enumerate(cf_candidates, 1)}
        sem_ranks = {isbn: rank for rank, (isbn, _) in enumerate(sem_candidates, 1)}

        cf_scores = dict(cf_candidates)
        sem_scores = dict(sem_candidates)

        all_isbns = set(cf_scores) | set(sem_scores)
        count_map = self._get_rating_count_map()

        scored: list[tuple[str, float]] = []
        for isbn in all_isbns:
            if isbn == seed_isbn:
                continue
            cf_s = cf_scores.get(isbn, 0.0)
            sem_s = sem_scores.get(isbn, 0.0)

            # Linear fusion
            fused = FUSION_ALPHA * cf_s + (1 - FUSION_ALPHA) * sem_s

            # Reciprocal rank fusion bonus for books appearing in both lists
            if isbn in cf_ranks and isbn in sem_ranks:
                k = 60  # smoothing constant
                rrf = 1.0 / (k + cf_ranks[isbn]) + 1.0 / (k + sem_ranks[isbn])
                fused += rrf

            # Mild popularity penalty
            rating_count = count_map.get(isbn, 1)
            fused = fused / math.log(1 + 0.3 * rating_count)

            scored.append((isbn, fused))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[Recommendation] = []
        for isbn, score in scored[:top_k]:
            results.append(self._build_recommendation(isbn, score))
        return results

    def _get_cf_candidates(self, seed_isbn: str) -> list[tuple[str, float]]:
        """Get top-N CF candidates as (isbn, similarity) pairs."""
        if not self.loader.has_cf:
            return []
        idx = self.loader.isbn_index.get(seed_isbn)
        if idx is None:
            return []

        row = self.loader.item_similarity.getrow(idx).toarray().flatten()
        top_indices = np.argsort(row)[::-1][:FUSION_CANDIDATE_POOL]

        candidates: list[tuple[str, float]] = []
        for col_idx in top_indices:
            sim = float(row[col_idx])
            if sim <= 0:
                break
            isbn = self.loader.index_isbn.get(int(col_idx))
            if isbn and isbn != seed_isbn:
                candidates.append((isbn, sim))
        return candidates

    def _get_semantic_candidates(self, seed_isbn: str) -> list[tuple[str, float]]:
        """Get top-N semantic candidates as (isbn, cosine_similarity) pairs."""
        if not self.loader.has_embeddings:
            return []
        try:
            seed_idx = self.loader.embedding_isbn_map.index(seed_isbn)
        except ValueError:
            return []

        seed_emb = self.loader.book_embeddings[seed_idx : seed_idx + 1]
        distances, indices = self.loader.faiss_index.search(
            seed_emb, FUSION_CANDIDATE_POOL + 1
        )

        candidates: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.loader.embedding_isbn_map):
                continue
            isbn = self.loader.embedding_isbn_map[idx]
            if isbn != seed_isbn:
                candidates.append((isbn, float(dist)))
        return candidates

    def _get_rating_count_map(self) -> dict[str, int]:
        stats = self.loader.book_stats
        if stats.empty or "rating_count" not in stats.columns:
            return {}
        return dict(zip(stats["ISBN"], stats["rating_count"]))


def _normalize_scores(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not candidates:
        return candidates
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(isbn, 1.0) for isbn, _ in candidates]
    return [(isbn, (s - lo) / (hi - lo)) for isbn, s in candidates]
