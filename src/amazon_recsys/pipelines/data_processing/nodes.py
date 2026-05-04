"""Data processing nodes for the Amazon Video Games recommendation system.

Each node is a pure function that takes Spark DataFrames (and optionally
parameters) and returns one or more Spark DataFrames.

No node calls .collect() on large data — only on small metadata/mappings.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, StringIndexerModel


def select_recommender_columns(df: DataFrame) -> DataFrame:
    """Select and rename columns needed for the collaborative-filtering pipeline.

    Args:
        df: Raw reviews DataFrame. Expected columns:
            user_id, parent_asin, rating, timestamp, verified_purchase, helpful_vote.

    Returns:
        DataFrame with columns:
            user_id, item_id, rating, timestamp, verified_purchase, helpful_vote.
    """
    return df.select(
        F.col("user_id"),
        F.col("parent_asin").alias("item_id"),
        F.col("rating").cast("float"),
        F.col("timestamp"),
        F.col("verified_purchase"),
        F.col("helpful_vote"),
    )


def clean_recommender_interactions(df: DataFrame) -> DataFrame:
    """Remove invalid rows from interaction data.

    Drops rows with nulls in user_id, item_id or rating.
    Filters ratings to the [1, 5] range.
    Replaces negative helpful_vote values with 0.

    Args:
        df: Selected interactions DataFrame.

    Returns:
        Cleaned DataFrame with the same schema.
    """
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df = df.filter((F.col("rating") >= 1) & (F.col("rating") <= 5))
    df = df.withColumn(
        "helpful_vote",
        F.when(F.col("helpful_vote") < 0, 0).otherwise(F.col("helpful_vote")),
    )
    return df


def deduplicate_user_item_interactions(df: DataFrame) -> DataFrame:
    """Keep only the most recent interaction per (user_id, item_id) pair.

    Uses a window partitioned by user_id + item_id ordered by timestamp
    descending, and keeps the first row (row_number == 1).

    Args:
        df: Cleaned interactions DataFrame.

    Returns:
        Deduplicated DataFrame with the same schema.
    """
    window = Window.partitionBy("user_id", "item_id").orderBy(
        F.col("timestamp").desc()
    )
    df = df.withColumn("_rn", F.row_number().over(window))
    df = df.filter(F.col("_rn") == 1).drop("_rn")
    return df


def filter_min_activity(
    df: DataFrame, min_user_interactions: int, min_item_interactions: int
) -> DataFrame:
    """Keep only users with >= N interactions and items with >= M interactions.

    Both filters are applied sequentially: first users, then items on the
    already-filtered set (so items that fall below the threshold after user
    filtering are also dropped).

    Args:
        df: Deduplicated interactions DataFrame.
        min_user_interactions: Minimum interactions a user must have.
        min_item_interactions: Minimum interactions an item must have.

    Returns:
        Filtered DataFrame with the same schema.
    """
    user_counts = df.groupBy("user_id").agg(F.count("*").alias("_ucnt"))
    active_users = user_counts.filter(
        F.col("_ucnt") >= min_user_interactions
    ).select("user_id")
    df = df.join(active_users, "user_id", "inner")

    item_counts = df.groupBy("item_id").agg(F.count("*").alias("_icnt"))
    active_items = item_counts.filter(
        F.col("_icnt") >= min_item_interactions
    ).select("item_id")
    df = df.join(active_items, "item_id", "inner")

    return df


def index_user_item_ids(
    df: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Map string user_id / item_id to sequential integer indices.

    Uses Spark's StringIndexer so that indices are assigned by descending
    frequency (most frequent → 0).

    Mappings are extracted directly from the transformed DataFrame instead of
    spark.createDataFrame with a Python list, which forces Spark to serialise
    data through Python workers and fails on Windows when the ``python``
    executable is not on PATH (e.g. the Microsoft Store stub).

    Args:
        df: Filtered interactions (must have user_id, item_id columns).

    Returns:
        A tuple of:
        - indexed_df:  same data with user_idx and item_idx (int) columns added.
        - user_mapping: DataFrame with columns [user_id, user_idx].
        - item_mapping: DataFrame with columns [item_id, item_idx].
    """
    user_indexer = StringIndexer(
        inputCol="user_id", outputCol="user_idx", handleInvalid="skip"
    )
    user_model: StringIndexerModel = user_indexer.fit(df)
    df = user_model.transform(df)

    item_indexer = StringIndexer(
        inputCol="item_id", outputCol="item_idx", handleInvalid="skip"
    )
    item_model: StringIndexerModel = item_indexer.fit(df)
    df = item_model.transform(df)

    df = df.withColumn("user_idx", F.col("user_idx").cast("int"))
    df = df.withColumn("item_idx", F.col("item_idx").cast("int"))

    user_mapping = df.select("user_id", "user_idx").distinct()
    item_mapping = df.select("item_id", "item_idx").distinct()

    df = df.select(
        "user_idx", "item_idx", "rating", "timestamp", "user_id", "item_id"
    )

    return df, user_mapping, item_mapping


def temporal_train_validation_test_split(
    df: DataFrame, split_params: dict
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split interactions into train / validation / test with temporal logic.

    For each user, interactions are ordered by timestamp descending:
        - Most recent  → test
        - Second most  → validation (if use_validation is True)
        - All others   → train

    Users with fewer than the required minimum interactions are dropped.

    Args:
        df: Indexed interactions (must have user_id, timestamp columns).
        split_params: Dictionary with:
            - use_validation (bool): whether to create a validation set.
            - timestamp_col (str): name of the timestamp column.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    use_validation = split_params.get("use_validation", True)
    timestamp_col = split_params.get("timestamp_col", "timestamp")

    window = Window.partitionBy("user_id").orderBy(
        F.col(timestamp_col).desc()
    )
    df = df.withColumn("_rank", F.row_number().over(window))

    test = df.filter(F.col("_rank") == 1).drop("_rank")

    if use_validation:
        validation = df.filter(F.col("_rank") == 2).drop("_rank")
        train = df.filter(F.col("_rank") >= 3).drop("_rank")
    else:
        spark = SparkSession.builder.getOrCreate()
        validation = spark.createDataFrame([], schema=df.drop("_rank").schema)
        train = df.filter(F.col("_rank") >= 2).drop("_rank")

    return train, validation, test
