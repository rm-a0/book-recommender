import sys
 
def _run_recommend(book_title: str, top_k: int = 10) -> int:
    from src.recommender.service import RecommenderService
 
    service = RecommenderService()
 
    matches = service.search_books(book_title, max_results=1)
    if not matches:
        print(f"No book found for title: {book_title!r}")
        return 1
 
    seed = matches[0]
    print(f"Using seed book: {seed.title} ({seed.isbn})")
    print()
 
    for result in service.recommend_all(seed.isbn, top_k=top_k):
        print(f"{result.strategy_label} [{result.strategy_name}]")
        if not result.recommendations:
            print("  No recommendations available.")
            print()
            continue
        for i, rec in enumerate(result.recommendations, 1):
            print(
                f"  {i}. {rec.title} - {rec.author} "
                f"(ISBN: {rec.isbn}, score={rec.score:.4f})"
            )
        print()
 
    return 0
 
 
def main() -> None:
    steps = {"ingest", "features", "enrich", "embeddings", "pipeline", "recommend"}
    args = sys.argv[1:]
 
    if not args or args[0] not in steps:
        print(__doc__)
        sys.exit(1)
 
    step = args[0]
 
    if step in ("ingest", "pipeline"):
        from src.pipeline.ingest import ingest_data
        ingest_data()
 
    if step in ("features", "pipeline"):
        from src.pipeline.features import build_all_features
        build_all_features()
 
    if step in ("enrich", "pipeline"):
        from src.pipeline.enrich import build_enriched_metadata
        build_enriched_metadata()
 
    if step in ("embeddings", "pipeline"):
        from src.pipeline.embeddings import build_all_embeddings
        build_all_embeddings()
 
    if step == "recommend":
        if len(args) < 2:
            print("Usage: python main.py recommend \"Book Title\"")
            sys.exit(1)
        query = " ".join(args[1:]).strip()
        sys.exit(_run_recommend(query))
 
if __name__ == "__main__":
    main()