from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from ..recommender.service import BookMatch, RecommenderService
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    BookSchema,
    HealthResponse,
    RecommendAllResponse,
    RecommendRequest,
    RecommendationSchema,
    StrategiesResponse,
    StrategyResultSchema,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = RecommenderService()
    yield

app = FastAPI(
    title="Book Recommender API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _service() -> RecommenderService:
    return app.state.service

def _book_to_schema(book: BookMatch) -> BookSchema:
    return BookSchema(
        isbn=book.isbn,
        title=book.title,
        author=book.author,
        rating_count=book.rating_count,
        bayesian_rating=book.bayesian_rating,
    )

def _resolve_seed(payload: RecommendRequest) -> BookMatch:
    service = _service()
    if payload.isbn:
        book = service.get_book(payload.isbn)
        if book is None:
            raise HTTPException(status_code=404, detail="Seed ISBN not found")
        return book

    assert payload.book_title is not None
    matches = service.search_books(payload.book_title, max_results=1)
    if not matches:
        raise HTTPException(status_code=404, detail="Seed title not found")
    return matches[0]

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = _service()
    return HealthResponse(
        status="ok",
        has_cf=service._loader.has_cf,
        has_embeddings=service._loader.has_embeddings,
        strategy_count=len(service.list_strategies()),
    )

@app.get("/strategies", response_model=StrategiesResponse)
def list_strategies() -> StrategiesResponse:
    return StrategiesResponse(strategies=_service().list_strategies())

@app.get("/books/search", response_model=list[BookSchema])
def search_books(
    q: str = Query(min_length=1),
    max_results: int = Query(default=10, ge=1, le=50),
) -> list[BookSchema]:
    books = _service().search_books(q, max_results=max_results)
    return [_book_to_schema(book) for book in books]

@app.get("/books/{isbn}", response_model=BookSchema)
def get_book(isbn: str) -> BookSchema:
    book = _service().get_book(isbn)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return _book_to_schema(book)

@app.post("/recommend", response_model=RecommendAllResponse)
def recommend_all(payload: RecommendRequest) -> RecommendAllResponse:
    service = _service()
    seed = _resolve_seed(payload)
    results = service.recommend_all(seed.isbn, top_k=payload.top_k)
    return RecommendAllResponse(
        seed_book=_book_to_schema(seed),
        strategies=[
            StrategyResultSchema(
                strategy_name=result.strategy_name,
                strategy_label=result.strategy_label,
                strategy_description=result.strategy_description,
                recommendations=[
                    RecommendationSchema(
                        isbn=rec.isbn,
                        title=rec.title,
                        author=rec.author,
                        score=rec.score,
                        strategy=rec.strategy,
                        description=rec.description,
                        subjects=rec.subjects,
                        rating_count=rec.rating_count,
                        bayesian_rating=rec.bayesian_rating,
                    )
                    for rec in result.recommendations
                ],
            )
            for result in results
        ],
    )

@app.post("/recommend/{strategy}", response_model=StrategyResultSchema)
def recommend_one(strategy: str, payload: RecommendRequest) -> StrategyResultSchema:
    service = _service()
    strategy_names = {s["name"] for s in service.list_strategies()}
    if strategy not in strategy_names:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    seed = _resolve_seed(payload)
    result = service.recommend(seed.isbn, strategy, top_k=payload.top_k)
    return StrategyResultSchema(
        strategy_name=result.strategy_name,
        strategy_label=result.strategy_label,
        strategy_description=result.strategy_description,
        recommendations=[
            RecommendationSchema(
                isbn=rec.isbn,
                title=rec.title,
                author=rec.author,
                score=rec.score,
                strategy=rec.strategy,
                description=rec.description,
                subjects=rec.subjects,
                rating_count=rec.rating_count,
                bayesian_rating=rec.bayesian_rating,
            )
            for rec in result.recommendations
        ],
    )