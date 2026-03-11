import sys

def main() -> None:
    steps = {"ingest", "features", "enrich", "embeddings", "pipeline"}
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


if __name__ == "__main__":
    main()
