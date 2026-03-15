from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from ..config import ARTIFACTS_PATH, ENRICHED_METADATA_PATH, PROCESSED_DATA_PATH

class ArtifactLoader:
    """Loads all artifacts from disk and provides access to them for strategies."""
    def __init__(
        self,
        artifacts_path: str = ARTIFACTS_PATH,
        enriched_path: str = ENRICHED_METADATA_PATH,
        processed_path: str = PROCESSED_DATA_PATH,
    ) -> None:
        self.artifacts_path = artifacts_path
        self.enriched_path = enriched_path
        self.processed_path = processed_path

        # Populated by load()
        self.isbn_index: dict[str, int] = {}
        self.index_isbn: dict[int, str] = {}
        self.item_similarity: csr_matrix | None = None
        self.book_stats: pd.DataFrame = pd.DataFrame()
        self.age_group_top_books: pd.DataFrame = pd.DataFrame()
        self.age_group_dominant: pd.DataFrame = pd.DataFrame()
        self.enriched_metadata: pd.DataFrame = pd.DataFrame()

        # Embedding artifacts (optional - not built by default pipeline)
        self.book_embeddings: np.ndarray | None = None
        self.faiss_index = None  # faiss.IndexFlatIP
        self.embedding_isbn_map: list[str] = []

        self._loaded = False

    def load(self) -> "ArtifactLoader":
        """Load all artifacts from disk. Idempotent."""
        if self._loaded:
            return self

        self._load_isbn_index()
        self._load_similarity_matrix()
        self._load_book_stats()
        self._load_age_group_data()
        self._load_enriched_metadata()
        self._load_embeddings()

        self._loaded = True
        return self

    # Loaders (tolerate missing files for flexibility)
    def _load_isbn_index(self) -> None:
        path = os.path.join(self.artifacts_path, "isbn_index.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            self.isbn_index = json.load(f)
        self.index_isbn = {v: k for k, v in self.isbn_index.items()}

    def _load_similarity_matrix(self) -> None:
        path = os.path.join(self.artifacts_path, "item_similarity.npz")
        if not os.path.exists(path):
            return
        self.item_similarity = load_npz(path)

    def _load_book_stats(self) -> None:
        path = os.path.join(self.artifacts_path, "book_stats.parquet")
        if not os.path.exists(path):
            return
        self.book_stats = pd.read_parquet(path)

    def _load_age_group_data(self) -> None:
        dom_path = os.path.join(self.artifacts_path, "age_group_dominant.parquet")
        top_path = os.path.join(self.artifacts_path, "age_group_top_books.parquet")
        if os.path.exists(dom_path):
            self.age_group_dominant = pd.read_parquet(dom_path)
        if os.path.exists(top_path):
            self.age_group_top_books = pd.read_parquet(top_path)

    def _load_enriched_metadata(self) -> None:
        if os.path.exists(self.enriched_path):
            self.enriched_metadata = pd.read_parquet(self.enriched_path)

    def _load_embeddings(self) -> None:
        emb_path = os.path.join(self.artifacts_path, "book_embeddings.npy")
        idx_path = os.path.join(self.artifacts_path, "faiss_index.bin")
        map_path = os.path.join(self.artifacts_path, "embedding_isbn_map.json")

        if os.path.exists(emb_path):
            self.book_embeddings = np.load(emb_path)

        if os.path.exists(idx_path):
            import faiss
            self.faiss_index = faiss.read_index(idx_path)

        if os.path.exists(map_path):
            with open(map_path) as f:
                self.embedding_isbn_map = json.load(f)

    # Helpers
    @property
    def has_cf(self) -> bool:
        return self.item_similarity is not None and len(self.isbn_index) > 0

    @property
    def has_embeddings(self) -> bool:
        return self.faiss_index is not None and len(self.embedding_isbn_map) > 0
