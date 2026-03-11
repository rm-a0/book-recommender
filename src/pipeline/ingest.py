import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from .schemas import get_book_schema, get_rating_schema, get_user_schema
from .clean import clean_ratings, clean_books, clean_users
from ..config import RAW_DATA_PATH, PROCESSED_DATA_PATH, BOOK_CSV, RATINGS_CSV, USERS_CSV
from .utils import create_spark_session, save_dataframe

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
    ratings_df = ratings_df.join(books_df.select("ISBN"), on="ISBN", how="inner")

    print(f"  Clean: {books_df.count():,} books, {ratings_df.count():,} ratings, {users_df.count():,} users")

    # Save to Parquet
    print("Saving processed Parquet...")
    os.makedirs(output_path, exist_ok=True)
    save_dataframe(books_df, output_path, "books.parquet")
    save_dataframe(ratings_df, output_path, "ratings.parquet")
    save_dataframe(users_df, output_path, "users.parquet")

    spark.stop()
    print("Done.")

if __name__ == "__main__":
    ingest_data()
