"""Shared ranking metrics for recommender pipelines."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_recall_at_k(
    recommendations: DataFrame, ground_truth: DataFrame, k: int
) -> float:
    """Compute Recall@K averaged across users."""
    relevant = ground_truth.select("user_idx", "item_idx").distinct()

    rel_count = relevant.groupBy("user_idx").agg(
        F.count("item_idx").alias("total_relevant")
    )

    top_k = recommendations.filter(F.col("rank") <= k)
    hits = top_k.join(relevant, ["user_idx", "item_idx"], "inner")
    hit_count = hits.groupBy("user_idx").agg(F.count("item_idx").alias("num_hits"))

    user_recall = rel_count.join(hit_count, "user_idx", "left").fillna(0)
    user_recall = user_recall.withColumn(
        "recall", F.col("num_hits") / F.col("total_relevant")
    )

    avg_recall = user_recall.select(F.avg("recall")).first()[0]
    return float(avg_recall) if avg_recall is not None else 0.0
