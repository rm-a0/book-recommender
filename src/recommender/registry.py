from __future__ import annotations
from .base import Recommendation, RecommendationStrategy
from .loader import ArtifactLoader

class StrategyRegistry:
    """Holds registered strategies and dispatches recommendation requests."""
    def __init__(self) -> None:
        self._strategies: dict[str, RecommendationStrategy] = {}

    def register(self, strategy: RecommendationStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> RecommendationStrategy:
        if name not in self._strategies:
            raise KeyError(f"Unknown strategy: {name!r}. Available: {list(self._strategies)}")
        return self._strategies[name]

    def list_strategies(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "label": s.label, "description": s.description}
            for s in self._strategies.values()
        ]

    def recommend_one(
        self, 
        strategy_name: str, 
        seed_isbn: str, 
        top_k: int = 10
    ) -> list[Recommendation]:
        return self.get(strategy_name).recommend(seed_isbn, top_k)

    def recommend_all(
        self, 
        seed_isbn: 
        str, 
        top_k: int = 10
    ) -> dict[str, list[Recommendation]]:
        results: dict[str, list[Recommendation]] = {}
        for name, strategy in self._strategies.items():
            results[name] = strategy.recommend(seed_isbn, top_k)
        return results

# Factory
def create_default_registry(loader: ArtifactLoader) -> StrategyRegistry:
    """Instantiate all strategies and return a fully-populated registry."""
    from .readers_also import ReadersAlso
    from .hidden_gems import HiddenGems
    from .similar_themes import SimilarThemes
    from .same_author import SameAuthor
    from .age_group import AgeGroup
    from .top_picks import TopPicks
 
    registry = StrategyRegistry()
    registry.register(ReadersAlso(loader))
    registry.register(HiddenGems(loader))
    registry.register(SimilarThemes(loader))
    registry.register(SameAuthor(loader))
    registry.register(AgeGroup(loader))
    registry.register(TopPicks(loader)) 
    return registry
