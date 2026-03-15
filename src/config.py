# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
ARTIFACTS_PATH = "artifacts"

# CSV filenames
BOOK_CSV = "Books.csv"
RATINGS_CSV = "Ratings.csv"
USERS_CSV = "Users.csv"

# Age bins for grouping users by age
AGE_BINS = [
    (13, 17, "13-17"), 
    (18, 24, "18-24"), 
    (25, 34, "25-34"),
    (35, 44, "35-44"), 
    (45, 54, "45-54"), 
    (55, 100, "55+"),
]

# Open Library API
OPENLIBRARY_ISBN_URL = "https://openlibrary.org/isbn/{isbn}.json"
OPENLIBRARY_WORKS_URL = "https://openlibrary.org{works_key}.json"
OPENLIBRARY_CACHE_PATH = "data/processed/openlibrary_cache.json"
ENRICHED_METADATA_PATH = "data/processed/book_metadata_enriched.parquet"
OPENLIBRARY_REQUEST_DELAY = 0.6  # seconds between API calls (100/min rate limit)

# Batch fetching optimization
OPENLIBRARY_BATCH_URL = "https://openlibrary.org/api/books"
BATCH_SIZE = 100
CONCURRENT_REQUESTS = 10
CACHE_SAVE_INTERVAL = 1000

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Recommender defaults
DEFAULT_TOP_K = 10
FUSION_ALPHA = 0.6  # weight for CF in top_picks fusion (1-alpha for semantic)
HIDDEN_GEM_CANDIDATE_POOL = 100  # CF candidates to consider for hidden gems reranking
FUSION_CANDIDATE_POOL = 50  # candidates per strategy for fusion
MIN_SEED_RATING = 7  # minimum rating to consider a user a "fan" of the seed book

# Collaborative-filtering feature build thresholds
CF_MIN_COMMON_RATERS = 2
CF_MIN_SIMILARITY = 0.0