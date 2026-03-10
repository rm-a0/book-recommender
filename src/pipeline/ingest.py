import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from .schemas import get_book_schema, get_rating_schema, get_user_schema
from .clean import clean_ratings, clean_books, clean_users

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

BOOK_CSV = "Books.csv"
RATINGS_CSV = "Ratings.csv"
USERS_CSV = "Users.csv"

def create_spark_session() -> SparkSession:
    """Creates and configures a SparkSession for local development."""
    return (
        SparkSession.builder
        .appName("BookRecommender-Ingest")
        .config("spark.driver.memory", "4g")
        .master("local[*]")
        .getOrCreate()
    )

def _read_csv(spark: SparkSession, file_path: str, schema) -> DataFrame:
    """Helper function to read a CSV file with a given schema."""
    return (
        spark.read
        .option("sep", ",")
        .option("encoding", "ISO-8859-1")
        .option("quote", '"')
        .option("escape", "\\")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(file_path)
    )

def _save_parquet(df: DataFrame, path: str, name: str) -> None:
    """Save dataframe to Parquet"""
    out = os.path.join(path, name)
    df.write.mode("overwrite").parquet(out)
    print(f"  Saved {name} -> {out}  ({df.count():,} rows)")

def ingest_data(data_path: str = RAW_DATA_PATH, output_path: str = PROCESSED_DATA_PATH):
    """Main function to ingest raw CSV data, clean it, and save as Parquet for downstream use."""
    spark = create_spark_session()

    # Load raw CSVs
    print("Loading raw CSVs...")
    books_df = _read_csv(spark, os.path.join(data_path, BOOK_CSV), get_book_schema())
    ratings_df = _read_csv(spark, os.path.join(data_path, RATINGS_CSV), get_rating_schema())
    users_df = _read_csv(spark, os.path.join(data_path, USERS_CSV), get_user_schema())

    print(f"  Raw: {books_df.count():,} books, {ratings_df.count():,} ratings, {users_df.count():,} users")

    # Clean data (order matters)
    print("Cleaning data...")
    ratings_df = clean_ratings(ratings_df)
    books_df = clean_books(books_df, ratings_df)
    users_df = clean_users(users_df)

    # Drop ratings for books that got filtered out during cleaning
    ratings_df = ratings_df.filter(
        F.col("ISBN").isin([r.ISBN for r in books_df.select("ISBN").distinct().collect()])
    )

    print(f"  Clean: {books_df.count():,} books, {ratings_df.count():,} ratings, {users_df.count():,} users")

    # Save to Parquet
    print("Saving processed Parquet...")
    os.makedirs(output_path, exist_ok=True)
    _save_parquet(books_df, output_path, "books.parquet")
    _save_parquet(ratings_df, output_path, "ratings.parquet")
    _save_parquet(users_df, output_path, "users.parquet")

    spark.stop()
    print("Done.")

if __name__ == "__main__":
    ingest_data()
