import json
import os
from scipy.sparse import csr_matrix, save_npz
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from ..config import PROCESSED_DATA_PATH, ARTIFACTS_PATH, AGE_BINS
from .utils import create_spark_session, save_dataframe


def load_processed_data(spark: SparkSession, path: str = PROCESSED_DATA_PATH):
    books = spark.read.parquet(f"{path}/books.parquet")
    ratings = spark.read.parquet(f"{path}/ratings.parquet")
    users = spark.read.parquet(f"{path}/users.parquet")
    return books, ratings, users

def compute_item_item_similarity(ratings_df: DataFrame, min_common_raters: int = 3) -> DataFrame:
    """Returns DataFrame[isbn_a, isbn_b, similarity, common_raters]."""
    # Create user - book - rating triples
    r = ratings_df.select(
        F.col("User-ID").alias("uid"),
        F.col("ISBN").alias("isbn"),
        F.col("Book-Rating").alias("rating")
    )

    # Self-join all pairs of books rated by the same user
    pairs = (
        r.alias("a")
        .join(r.alias("b"), on="uid")
        .filter(F.col("a.isbn") < F.col("b.isbn")) # avoid duplicates
        .select(
            F.col("a.isbn").alias("isbn_a"),
            F.col("b.isbn").alias("isbn_b"),
            F.col("a.rating").alias("ra"),
            F.col("b.rating").alias("rb")
        )
    )

    # Aggregate to compute dot product, norms and rater count
    sim = (
        pairs.groupBy("isbn_a", "isbn_b")
        .agg(
            F.sum(F.col("ra") * F.col("rb")).alias("dot_product"),
            F.sqrt(F.sum(F.col("ra") ** 2)).alias("norm_a"),
            F.sqrt(F.sum(F.col("rb") ** 2)).alias("norm_b"),
            F.count("*").alias("common_raters")
        )
        .filter(F.col("common_raters") >= min_common_raters)
        .withColumn(
            "similarity",
            F.col("dot_product") / (F.col("norm_a") * F.col("norm_b"))
        )
        .select("isbn_a", "isbn_b", "similarity", "common_raters")
    )

    return sim

def similarity_to_sparse(sim_df: DataFrame, isbn_index: dict[str, int]) -> csr_matrix:
    """Pack pairwise similarities into a symmetric scipy CSR matrix for O(1) row lookup."""
    rows = sim_df.collect()
    n = len(isbn_index)
    row_idx, col_idx, data = [], [], []
    for r in rows:
        i, j = isbn_index.get(r.isbn_a), isbn_index.get(r.isbn_b)
        if i is None or j is None:
            continue
        # Fill both (i, j) and (j, i) for symmetry
        row_idx.extend([i, j])
        col_idx.extend([j, i])
        data.extend([r.similarity, r.similarity])
    return csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

def compute_book_stats(books_df: DataFrame, ratings_df: DataFrame) -> DataFrame:
    """Join books with per-ISBN rating_count, rating_mean, bayesian_rating."""
    # Compute rating stats per book
    stats = ratings_df.groupBy("ISBN").agg(
        F.count("*").alias("rating_count"),
        F.avg("Book-Rating").alias("rating_mean"),
    )
    global_mean = ratings_df.select(F.avg("Book-Rating")).first()[0]

    # Bayesian average: (v / (v + m)) * R + (m / (v + m)) * C
    # where R = rating_mean, v = rating_count, C = global_mean,
    # and m = minimum votes to be considered credible (set to 25th percentile).
    m = float(stats.approxQuantile("rating_count", [0.25], 0.05)[0])
    stats = stats.withColumn(
        "bayesian_rating",
        (F.col("rating_count") / (F.col("rating_count") + F.lit(m))) * F.col("rating_mean")
        + (F.lit(m) / (F.col("rating_count") + F.lit(m))) * F.lit(global_mean),
    )
    return books_df.join(stats, on="ISBN", how="inner")

def _age_group_expr(col: str = "Age") -> F.Column:
    """Maps age to age group label based on predefined bins."""
    expr = F.lit(None).cast("string")
    for lo, hi, label in reversed(AGE_BINS):
        expr = F.when((F.col(col) >= lo) & (F.col(col) <= hi), F.lit(label)).otherwise(expr)
    return expr

def _ratings_with_age_group(ratings_df: DataFrame, users_df: DataFrame) -> DataFrame:
    """Join ratings with user ages and assign age group."""
    return (
        ratings_df
        .join(users_df.select("User-ID", "Age"), on="User-ID", how="inner")
        .filter(F.col("Age").isNotNull())
        .withColumn("age_group", _age_group_expr())
        .filter(F.col("age_group").isNotNull())
    )

def compute_age_group_dominant(ratings_df: DataFrame, users_df: DataFrame) -> DataFrame:
    """For each book, which age bracket contributed the most raters."""
    rated = _ratings_with_age_group(ratings_df, users_df)
    counts = rated.groupBy("ISBN", "age_group").agg(F.count("*").alias("age_group_count"))
    w = Window.partitionBy("ISBN").orderBy(F.col("age_group_count").desc())
    return (
        counts.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumnRenamed("age_group", "dominant_age_group")
    )

def compute_age_group_top_books(
    ratings_df: DataFrame, users_df: DataFrame, book_stats_df: DataFrame
) -> DataFrame:
    """For each age group, list all books rated in that group with their bayesian rating."""
    rated = _ratings_with_age_group(ratings_df, users_df)
    books_per_group = rated.select("age_group", "ISBN").distinct()
    return books_per_group.join(
        book_stats_df.select("ISBN", "bayesian_rating"), on="ISBN", how="inner"
    )

def build_all_features(
    data_path: str = PROCESSED_DATA_PATH,
    artifacts_path: str = ARTIFACTS_PATH,
):
    """Main function to build all features and save to artifacts/ for downstream use."""
    spark = create_spark_session()
    os.makedirs(artifacts_path, exist_ok=True)
    books, ratings, users = load_processed_data(spark, data_path)

    # Create ISBN index for matrix rows/columns
    print("Building ISBN index...")
    isbn_list = sorted([r.ISBN for r in books.select("ISBN").distinct().collect()])
    isbn_index = {isbn: i for i, isbn in enumerate(isbn_list)}
    idx_path = os.path.join(artifacts_path, "isbn_index.json")
    with open(idx_path, "w") as f:
        json.dump(isbn_index, f)
    print(f"  Saved {idx_path}  ({len(isbn_index):,} unique ISBNs)")

    # Item-item similarity matrix
    print("Computing item-item similarity...")
    sim_df = compute_item_item_similarity(ratings)
    sparse_sim = similarity_to_sparse(sim_df, isbn_index)
    sim_path = os.path.join(artifacts_path, "item_similarity.npz")
    save_npz(sim_path, sparse_sim)
    print(f"  Saved {sim_path}  ({sparse_sim.nnz:,} non-zero entries)")

    # Book stats - rating count, mean, bayesian avg
    print("Computing book stats...")
    book_stats = compute_book_stats(books, ratings)
    save_dataframe(book_stats, artifacts_path, "book_stats.parquet")

    # Age group -  dominant group per book + top books per group
    print("Computing age group stats...")
    age_dom = compute_age_group_dominant(ratings, users)
    save_dataframe(age_dom, artifacts_path, "age_group_dominant.parquet")

    # For each age group, list all books rated in that group with their bayesian rating
    age_top = compute_age_group_top_books(ratings, users, book_stats)
    save_dataframe(age_top, artifacts_path, "age_group_top_books.parquet")

    spark.stop()
    print("Done.")

if __name__ == "__main__":
    build_all_features()