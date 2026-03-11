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
    (55, 200, "55+"),
]