import asyncio
import json
import os
import httpx
import pandas as pd
from ..config import (
    ENRICHED_METADATA_PATH,
    OPENLIBRARY_CACHE_PATH,
    PROCESSED_DATA_PATH,
    OPENLIBRARY_BATCH_URL,
    BATCH_SIZE,
    CONCURRENT_REQUESTS,
    CACHE_SAVE_INTERVAL,
)

def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

async def fetch_batch(client: httpx.AsyncClient, isbns: list[str]) -> dict:
    """Fetch metadata for up to 100 ISBNs in one request."""
    bibkeys = ",".join(f"ISBN:{isbn}" for isbn in isbns)

    params = {
        "bibkeys": bibkeys,
        "format": "json",
        "jscmd": "data",
    }

    try:
        resp = await client.get(OPENLIBRARY_BATCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {isbn: {"description": "", "subjects": []} for isbn in isbns}

    result = {}

    for isbn in isbns:
        key = f"ISBN:{isbn}"
        entry = data.get(key, {})

        description = ""
        subjects = []

        desc = entry.get("description")
        if isinstance(desc, dict):
            description = desc.get("value", "")
        elif isinstance(desc, str):
            description = desc

        if "subjects" in entry:
            subjects = [s.get("name") for s in entry["subjects"] if "name" in s]

        result[isbn] = {
            "description": description or "",
            "subjects": subjects or [],
        }

    return result


async def batch_fetch_metadata_async(
    isbn_list: list[str],
    cache_path: str,
) -> dict:

    cache = _load_cache(cache_path)

    to_fetch = [isbn for isbn in isbn_list if isbn not in cache]

    if not to_fetch:
        print(f"All {len(isbn_list):,} ISBNs already cached.")
        return cache

    print(f"Fetching {len(to_fetch):,} ISBNs from Open Library...")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(timeout=20) as client:

        async def process_batch(batch):
            async with semaphore:
                return await fetch_batch(client, batch)

        batches = [
            to_fetch[i:i + BATCH_SIZE]
            for i in range(0, len(to_fetch), BATCH_SIZE)
        ]

        processed = 0

        for i in range(0, len(batches), CONCURRENT_REQUESTS):

            chunk = batches[i:i + CONCURRENT_REQUESTS]

            results = await asyncio.gather(
                *[process_batch(batch) for batch in chunk]
            )

            for batch_result in results:
                cache.update(batch_result)

            processed += sum(len(b) for b in chunk)

            if processed % CACHE_SAVE_INTERVAL == 0:
                _save_cache(cache, cache_path)

            print(f"Progress: {processed:,}/{len(to_fetch):,}")

    _save_cache(cache, cache_path)

    print(f"Cache now contains {len(cache):,} books")

    return cache

def build_enriched_metadata(
    data_path: str = PROCESSED_DATA_PATH,
    output_path: str = ENRICHED_METADATA_PATH,
    cache_path: str = OPENLIBRARY_CACHE_PATH,
):

    books_parquet = os.path.join(data_path, "books.parquet")
    books_df = pd.read_parquet(books_parquet)

    isbn_list = books_df["ISBN"].tolist()

    print(f"Enriching {len(isbn_list):,} books...")

    metadata = asyncio.run(
        batch_fetch_metadata_async(isbn_list, cache_path)
    )

    # FAST lookup instead of dataframe scan
    books_lookup = books_df.set_index("ISBN").to_dict("index")

    rows = []

    for isbn in isbn_list:

        book_row = books_lookup.get(isbn, {})

        title = book_row.get("Book-Title", "")
        author = book_row.get("Book-Author", "")

        meta = metadata.get(isbn, {"description": "", "subjects": []})

        description = meta["description"]
        subjects = meta["subjects"]

        if not description:
            description = f"{title} by {author}" if author else title

        rows.append({
            "ISBN": isbn,
            "Book-Title": title,
            "Book-Author": author,
            "description": description,
            "subjects": ", ".join(subjects[:20]),
        })

    enriched = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    enriched.to_parquet(output_path, index=False)

    with_desc = enriched[enriched["description"].str.len() > 20]
    coverage = len(with_desc) / len(enriched) * 100

    print(
        f"Saved {output_path} "
        f"({len(enriched):,} books, {coverage:.0f}% with real descriptions)"
    )


if __name__ == "__main__":
    build_enriched_metadata()