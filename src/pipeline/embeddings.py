import json
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from ..config import (
    ARTIFACTS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    ENRICHED_METADATA_PATH,
)

def _build_text(row: pd.Series) -> str:
    """Construct a single representative text string per book for embedding.

    Combines title, author, description, and subjects into one sentence so the
    model can encode the full semantic content of the book in a single vector.
    """
    title = row.get("Book-Title", "")
    author = row.get("Book-Author", "")
    description = row.get("description", "")
    subjects = row.get("subjects", "")

    # Build header first so it's available for the description dedup check
    header = f"{title} by {author}" if author else title

    parts = []
    if header:
        parts.append(header)

    # Skip description when it is the fallback "Title by Author" string
    if description and description != header:
        parts.append(description)

    if subjects:
        parts.append(f"Genres: {subjects}")

    return ". ".join(parts)

def _generate_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Encode a list of texts into a (N, dim) float32 array.

    Uses a SentenceTransformer model and L2-normalises the output so that
    inner product is equivalent to cosine similarity, which is what FAISS IndexFlatIP expects.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # L2-normalise embeddings to unit length for cosine similarity search in FAISS
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against zero vectors from empty texts
    embeddings = embeddings / norms

    return embeddings.astype(np.float32)

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS exact inner-product index from pre-normalised embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def build_all_embeddings(
    enriched_path: str = ENRICHED_METADATA_PATH,
    artifacts_path: str = ARTIFACTS_PATH,
) -> None:
    """Embed enriched book metadata and build a FAISS index for similarity search."""
    print("Loading enriched metadata...")
    df = pd.read_parquet(enriched_path)
    print(f"  {len(df):,} books loaded")

    # Build one text string per book combining all available metadata fields
    texts = df.apply(_build_text, axis=1).tolist()
    isbn_list = df["ISBN"].tolist()

    print(f"Generating embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings = _generate_embeddings(texts)

    assert embeddings.shape == (len(texts), EMBEDDING_DIM), (
        f"Embedding shape mismatch: expected ({len(texts)}, {EMBEDDING_DIM}), "
        f"got {embeddings.shape}"
    )

    print("Building FAISS index...")
    index = _build_faiss_index(embeddings)

    # Save all three artifacts
    print("Saving artifacts...")
    os.makedirs(artifacts_path, exist_ok=True)

    emb_path = os.path.join(artifacts_path, "book_embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"  Saved {emb_path}  ({embeddings.shape})")

    idx_path = os.path.join(artifacts_path, "faiss_index.bin")
    faiss.write_index(index, idx_path)
    print(f"  Saved {idx_path}  ({index.ntotal:,} vectors, dim={index.d})")

    map_path = os.path.join(artifacts_path, "embedding_isbn_map.json")
    with open(map_path, "w") as f:
        json.dump(isbn_list, f)
    print(f"  Saved {map_path}  ({len(isbn_list):,} ISBNs)")

    print("Done.")

if __name__ == "__main__":
    build_all_embeddings()