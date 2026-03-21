import math
import numpy as np
from rapidfuzz import fuzz
from ..config import (
    FUSION_ALPHA,
    FUSION_CANDIDATE_POOL,
    MIN_BAYESIAN_RATING_FLOOR,
    MIN_CANDIDATE_RATING_COUNT,
    EDITION_SIMILARITY_THRESHOLD,
)
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader


class TopPicks(RecommendationStrategy):
    name = "top_picks"
    label = "Top Picks For You"
    description = "Best recommendations combining collaborative and semantic signals."

    def __init__(self, loader: ArtifactLoader) -> None:
        super().__init__(loader)

    def recommend(self, seed_isbn: str, top_k: int = 10) -> list[Recommendation]:
        cf_candidates  = self._get_cf_candidates(seed_isbn)
        sem_candidates = self._get_semantic_candidates(seed_isbn)

        if not cf_candidates and not sem_candidates:
            return []

        # Build stat lookups once (avoid repeated DataFrame scans)
        stats = self.loader.book_stats
        rating_count_map: dict[str, int]   = {}
        bayesian_map: dict[str, float] = {}
        title_map: dict[str, str]   = {}
        if not stats.empty:
            rating_count_map = dict(zip(stats["ISBN"], stats["rating_count"]))
            bayesian_map = dict(zip(stats["ISBN"], stats["bayesian_rating"]))
            if "Book-Title" in stats.columns:
                title_map = dict(zip(stats["ISBN"], stats["Book-Title"].fillna("").str.lower()))

        seed_title = title_map.get(seed_isbn, "")

        # Pre-filter candidates: drop books with too few ratings (noisy CF signal)
        def _keep(isbn: str) -> bool:
            if isbn == seed_isbn:
                return False
            if rating_count_map.get(isbn, 0) < MIN_CANDIDATE_RATING_COUNT:
                return False
            # Drop likely editions of the seed book
            if seed_title:
                candidate_title = title_map.get(isbn, "")
                if candidate_title and fuzz.ratio(candidate_title, seed_title) >= EDITION_SIMILARITY_THRESHOLD:
                    return False
            return True

        cf_candidates = [(isbn, s) for isbn, s in cf_candidates  if _keep(isbn)]
        sem_candidates = [(isbn, s) for isbn, s in sem_candidates if _keep(isbn)]

        # Normalize scores to [0, 1] within each list
        cf_candidates = _normalize_scores(cf_candidates)
        sem_candidates = _normalize_scores(sem_candidates)

        # Rank maps for Reciprocal Rank Fusion bonus
        cf_ranks = {isbn: rank for rank, (isbn, _) in enumerate(cf_candidates,  1)}
        sem_ranks = {isbn: rank for rank, (isbn, _) in enumerate(sem_candidates, 1)}

        cf_scores  = dict(cf_candidates)
        sem_scores = dict(sem_candidates)

        all_isbns = set(cf_scores) | set(sem_scores)

        scored: list[tuple[str, float]] = []
        for isbn in all_isbns:
            cf_s  = cf_scores.get(isbn,  0.0)
            sem_s = sem_scores.get(isbn, 0.0)

            # Linear fusion
            fused = FUSION_ALPHA * cf_s + (1 - FUSION_ALPHA) * sem_s

            # RRF bonus for books appearing in both lists
            if isbn in cf_ranks and isbn in sem_ranks:
                k = 60
                fused += 1.0 / (k + cf_ranks[isbn]) + 1.0 / (k + sem_ranks[isbn])

            # Mild popularity penalty
            rc = rating_count_map.get(isbn, 1)
            fused = fused / math.log(1 + 0.3 * rc)

            scored.append((isbn, fused))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Quality floor: drop books below minimum Bayesian rating
        scored = [
            (isbn, s) for isbn, s in scored
            if bayesian_map.get(isbn, 0.0) >= MIN_BAYESIAN_RATING_FLOOR
        ]

        results: list[Recommendation] = []
        for isbn, score in scored[:top_k]:
            results.append(self._build_recommendation(isbn, score))
        return results

    def _get_cf_candidates(self, seed_isbn: str) -> list[tuple[str, float]]:
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
        if not self.loader.has_embeddings:
            return []
        try:
            seed_idx = self.loader.embedding_isbn_map.index(seed_isbn)
        except ValueError:
            return []

        seed_emb = self.loader.book_embeddings[seed_idx : seed_idx + 1]
        distances, indices = self.loader.faiss_index.search(seed_emb, FUSION_CANDIDATE_POOL + 1)

        candidates: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.loader.embedding_isbn_map):
                continue
            isbn = self.loader.embedding_isbn_map[idx]
            if isbn != seed_isbn:
                candidates.append((isbn, float(dist)))
        return candidates


def _normalize_scores(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not candidates:
        return candidates
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(isbn, 1.0) for isbn, _ in candidates]
    return [(isbn, (s - lo) / (hi - lo)) for isbn, s in candidates]