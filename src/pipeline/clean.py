from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def clean_ratings(
    ratings_df: DataFrame,
    min_rating: int = 1,
    max_rating: int = 10,
    min_ratings_per_user: int = 3,
) -> DataFrame:
    """Full ratings cleaning pipeline."""
    ratings_df = drop_null_columns(ratings_df, ["User-ID", "ISBN", "Book-Rating"])
    ratings_df = drop_duplicate_ratings(ratings_df, ["User-ID", "ISBN"])
    ratings_df = filter_explicit_ratings(ratings_df, min_rating, max_rating)
    ratings_df = filter_min_ratings_per_user(ratings_df, min_ratings_per_user)
    return ratings_df

def clean_books(
    books_df: DataFrame,
    ratings_df: DataFrame,
    min_ratings_per_book: int = 5,
) -> DataFrame:
    """Full books cleaning pipeline. Must be called AFTER clean_ratings."""
    books_df = drop_null_columns(books_df, ["ISBN", "Book-Title"])
    books_df = normalize_titles(books_df)
    books_df = normalize_authors(books_df)
    books_df = clean_year_of_publication(books_df)
    books_df = drop_books_without_ratings(books_df, ratings_df)
    books_df = filter_min_ratings_per_book(books_df, ratings_df, min_ratings_per_book)
    books_df = deduplicate_editions(books_df)
    return books_df

def clean_users(users_df: DataFrame, min_age: int = 13, max_age: int = 100) -> DataFrame:
    """Full users cleaning pipeline."""
    users_df = drop_null_columns(users_df, ["User-ID"])
    users_df = clamp_age(users_df, min_age, max_age)
    return users_df

def drop_null_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    return df.dropna(subset=columns)

def drop_duplicate_ratings(df: DataFrame, columns: list[str]) -> DataFrame:
    return df.dropDuplicates(subset=columns)

def filter_explicit_ratings(
    df: DataFrame, 
    min_rating: int = 1, 
    max_rating: int = 10, 
    col: str = "Book-Rating"
) -> DataFrame:
    """Keep only explicit ratings (>=1)."""
    return df.filter((F.col(col) >= min_rating) & (F.col(col) <= max_rating))

def filter_min_ratings_per_user(
    df: DataFrame, 
    min_count: int = 3, 
    user_col: str = "User-ID"
) -> DataFrame:
    """Drop users with fewer than min_count ratings """
    user_counts = df.groupBy(user_col).count().filter(F.col("count") >= min_count)
    return df.join(user_counts.select(user_col), on=user_col, how="inner")

def normalize_titles(df: DataFrame, col: str = "Book-Title") -> DataFrame:
    """Lowercase, strip whitespace, normalize unicode."""
    return df.withColumn(col, F.trim(F.lower(F.col(col))))

def normalize_authors(df: DataFrame, col: str = "Book-Author") -> DataFrame:
    """Lowercase, strip whitespace, normalize unicode for author matching."""
    return df.withColumn(col, F.trim(F.lower(F.col(col))))

def clean_year_of_publication(df: DataFrame, col: str = "Year-Of-Publication") -> DataFrame:
    """Cast to int, null out garbage values (0, negative, future years, non-numeric)."""
    df = df.withColumn(col, F.expr(f"try_cast(`{col}` as int)"))
    df = df.withColumn(
        col,
        F.when((F.col(col) >= 1500) & (F.col(col) <= 2025), F.col(col)).otherwise(None),
    )
    return df

def drop_books_without_ratings(
    books_df: DataFrame, 
    ratings_df: DataFrame, 
    isbn_col: str = "ISBN"
) -> DataFrame:
    """Drop books with no ratings"""
    rated_isbns = ratings_df.select(isbn_col).distinct()
    return books_df.join(rated_isbns, on=isbn_col, how="inner")

def filter_min_ratings_per_book(
    books_df: DataFrame,
    ratings_df: DataFrame,
    min_count: int = 5,
    isbn_col: str = "ISBN",
) -> DataFrame:
    """Drop books with fewer than min_count ratings."""
    book_counts = ratings_df.groupBy(isbn_col).count().filter(F.col("count") >= min_count)
    return books_df.join(book_counts.select(isbn_col), on=isbn_col, how="inner")

def deduplicate_editions(
    df: DataFrame, 
    title_col: str = "Book-Title", 
    author_col: str = "Book-Author"
) -> DataFrame:
    """
    When multiple ISBNs share the same title + author, keep the one 
    with the lowest ISBN. This collapses different editions into one entry.
    """
    w = Window.partitionBy(title_col, author_col).orderBy("ISBN")
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

def clamp_age(df: DataFrame, min_age: int = 13, max_age: int = 100, col: str = "Age") -> DataFrame:
    """Null out ages outside the plausible range."""
    return df.withColumn(
        col,
        F.when((F.col(col) >= min_age) & (F.col(col) <= max_age), F.col(col)).otherwise(None),
    )
