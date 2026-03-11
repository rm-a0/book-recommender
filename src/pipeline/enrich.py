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
    """Load the ISBN metadata cache from disk, or return an empty dict."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict, cache_path: str) -> None:
    """Persist the ISBN metadata cache to disk."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

async def _fetch_batch(client: httpx.AsyncClient, isbns: list[str]) -> dict:
    """Fetch metadata for up to 100 ISBNs in one OpenLibrary request.

    Returns a dict keyed by ISBN with ``description`` and ``subjects`` fields.
    Falls back to empty values on any network or parse error so the caller
    never has to handle exceptions from individual batches.
    """
    bibkeys = ",".join(f"ISBN:{isbn}" for isbn in isbns)
    params = {"bibkeys": bibkeys, "format": "json", "jscmd": "data"}

    try:
        resp = await client.get(OPENLIBRARY_BATCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {isbn: {"description": "", "subjects": []} for isbn in isbns}

    result = {}
    for isbn in isbns:
        entry = data.get(f"ISBN:{isbn}", {})

        # description can be a plain string or {"type": ..., "value": ...}
        desc = entry.get("description", "")
        description = desc.get("value", "") if isinstance(desc, dict) else desc

        subjects = [s["name"] for s in entry.get("subjects", []) if "name" in s]

        result[isbn] = {
            "description": description or "",
            "subjects": subjects,
        }

    return result

async def _batch_fetch_metadata_async(
    isbn_list: list[str],
    cache_path: str,
) -> dict:
    """Fetch OpenLibrary metadata for every ISBN not already in the cache.

    Batches requests to respect the API's 100-key limit per call and fires
    ``CONCURRENT_REQUESTS`` batches simultaneously. The cache is saved to disk
    every ``CACHE_SAVE_INTERVAL`` ISBNs processed so progress survives crashes.
    """
    cache = _load_cache(cache_path)

    to_fetch = [isbn for isbn in isbn_list if isbn not in cache]
    if not to_fetch:
        print(f"  All {len(isbn_list):,} ISBNs already cached.")
        return cache

    print(f"  Fetching {len(to_fetch):,} ISBNs from OpenLibrary...")

    batches = [
        to_fetch[i : i + BATCH_SIZE]
        for i in range(0, len(to_fetch), BATCH_SIZE)
    ]

    processed = 0
    last_saved = 0  # tracks ISBNs processed at the last cache flush

    async with httpx.AsyncClient(timeout=20) as client:
        for i in range(0, len(batches), CONCURRENT_REQUESTS):
            chunk = batches[i : i + CONCURRENT_REQUESTS]

            # Fire up to CONCURRENT_REQUESTS batches simultaneously
            results = await asyncio.gather(
                *[_fetch_batch(client, batch) for batch in chunk]
            )

            for batch_result in results:
                cache.update(batch_result)

            processed += sum(len(b) for b in chunk)
            print(f"  Progress: {processed:,}/{len(to_fetch):,}")

            # Flush to disk periodically so a crash doesn't lose all progress
            if processed - last_saved >= CACHE_SAVE_INTERVAL:
                _save_cache(cache, cache_path)
                last_saved = processed

    _save_cache(cache, cache_path)
    print(f"  Cache now contains {len(cache):,} entries")

    return cache

def build_enriched_metadata(
    data_path: str = PROCESSED_DATA_PATH,
    output_path: str = ENRICHED_METADATA_PATH,
    cache_path: str = OPENLIBRARY_CACHE_PATH,
) -> None:
    """Enrich books with OpenLibrary descriptions and subjects, then save to Parquet."""
    books_df = pd.read_parquet(os.path.join(data_path, "books.parquet"))
    isbn_list = books_df["ISBN"].tolist()
    print(f"Enriching {len(isbn_list):,} books...")

    # Fetch metadata (async, cached)
    metadata = asyncio.run(_batch_fetch_metadata_async(isbn_list, cache_path))

    # Build enriched rows using a dict lookup instead of per-row DataFrame scans
    books_lookup = books_df.set_index("ISBN").to_dict("index")
    rows = []
    for isbn in isbn_list:
        book   = books_lookup.get(isbn, {})
        title  = book.get("Book-Title", "")
        author = book.get("Book-Author", "")
        meta   = metadata.get(isbn, {"description": "", "subjects": []})

        description = meta["description"]
        subjects    = meta["subjects"]

        # Fall back to "Title by Author" when no real description was returned
        if not description:
            description = f"{title} by {author}" if author else title

        rows.append({
            "ISBN":        isbn,
            "Book-Title":  title,
            "Book-Author": author,
            "description": description,
            "subjects":    ", ".join(subjects[:20]),
        })

    enriched = pd.DataFrame(rows)

    # Save
    print("Saving enriched metadata...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    enriched.to_parquet(output_path, index=False)

    # A description is meaningfully longer than the "Title by Author" fallback
    min_real_desc_len = 40
    coverage = (enriched["description"].str.len() > min_real_desc_len).sum() / len(enriched) * 100
    print(f"  Saved {output_path}  ({len(enriched):,} books, {coverage:.0f}% with real descriptions)")
    print("Done.")


if __name__ == "__main__":
    build_enriched_metadata()
