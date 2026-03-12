from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def get_book_schema() -> StructType:
    """Defines the schema for the book dataset."""
    return StructType([
        StructField("ISBN", StringType(), False),
        StructField("Book-Title", StringType(), True),
        StructField("Book-Author", StringType(), True),
        StructField("Year-Of-Publication", StringType(), True),
        StructField("Publisher", StringType(), True),
        StructField("Image-URL-S", StringType(), True),
        StructField("Image-URL-M", StringType(), True),
        StructField("Image-URL-L", StringType(), True),
    ])

def get_rating_schema() -> StructType:
    """Defines the schema for the ratings dataset."""
    return StructType([
        StructField("User-ID", IntegerType(), False),
        StructField("ISBN", StringType(), False),
        StructField("Book-Rating", IntegerType(), True),
    ])

def get_user_schema() -> StructType:
    """Defines the schema for the users dataset."""
    return StructType([
        StructField("User-ID", IntegerType(), False),
        StructField("Location", StringType(), True),
        StructField("Age", StringType(), True),
    ])