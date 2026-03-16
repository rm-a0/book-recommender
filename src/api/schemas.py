from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

class BookSchema(BaseModel):
    isbn: str
    title: str
    author: str
    rating_count: int
    bayesian_rating: float

class RecommendationSchema(BaseModel):
    isbn: str
    title: str
    author: str
    score: float
    strategy: str
    description: str = ""
    subjects: str = ""
    rating_count: int = 0
    bayesian_rating: float = 0.0

class StrategyResultSchema(BaseModel):
    strategy_name: str
    strategy_label: str
    strategy_description: str
    recommendations: list[RecommendationSchema]

class RecommendRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    book_title: str | None = Field(default=None, min_length=1)
    isbn: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def validate_seed(self) -> "RecommendRequest":
        if self.book_title or self.isbn:
            return self
        raise ValueError("Provide either book_title or isbn")

class RecommendAllResponse(BaseModel):
    seed_book: BookSchema
    strategies: list[StrategyResultSchema]

class StrategiesResponse(BaseModel):
    strategies: list[dict[str, str]]

class HealthResponse(BaseModel):
    status: str
    has_cf: bool
    has_embeddings: bool
    strategy_count: int