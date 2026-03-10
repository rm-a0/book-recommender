import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from .schemas import get_book_schema, get_rating_schema, get_user_schema

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

BOOK_CSV = "Books.csv"
RATINGS_CSV = "Ratings.csv"
USERS_CSV = "Users.csv"

def create_spark_session() -> SparkSession:
    """Creates and configures a SparkSession for local development."""
    session = (
        SparkSession.builder
        .appName("BookRecommender")
        .config("spark.driver.memory", "4g")
        .master("local[*]")
        .getOrCreate()
    )
    return session

def _read_csv(spark: SparkSession, file_path: str, schema) -> DataFrame:
    """Helper function to read a CSV file with a given schema."""
    return (
        spark.read
        .option("sep", ";")
        .option("encoding", "ISO-8859-1")
        .option("quote", '"')
        .option("escape", '"')
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(file_path)
    )

def ingest_data(data_path: str = RAW_DATA_PATH, output_path: str = PROCESSED_DATA_PATH):
    spark = create_spark_session()

    # Load CSV datasets with defined schemas
    books_df = _read_csv(spark, os.path.join(data_path, BOOK_CSV), get_book_schema())
    ratings_df = _read_csv(spark, os.path.join(data_path, RATINGS_CSV), get_rating_schema())
    users_df = _read_csv(spark, os.path.join(data_path, USERS_CSV, get_user_schema()))

    # Clean and preprocess data

