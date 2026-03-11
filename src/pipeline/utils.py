import os
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

def create_spark_session(name: str="BookRecommender") -> SparkSession:
    """Creates and configures a SparkSession for local development."""
    return (
        SparkSession.builder
        .appName(name)
        .config("spark.driver.memory", "4g")
        .master("local[*]")
        .getOrCreate()
    )

def save_dataframe(df: DataFrame, path: str, name: str) -> None:
    """Save a Spark DataFrame to Parquet format."""
    out = os.path.join(path, name)
    df.write.mode("overwrite").parquet(path)
    print(f"  Saved {name} -> {out}  ({df.count():,} rows)")