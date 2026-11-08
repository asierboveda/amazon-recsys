"""Shared warm-start evaluation protocol for the four recommenders.

The split is retrospective per user. All models rank the same positive-training
catalogue and are scored against exactly the same eligible held-out rows.
"""

from pyspark.sql import functions as F


def prepare_experiment(train, validation, test, params):
    """Build positive training data, eligible targets, seen history and counts.

    Exclusion counts are mutually exclusive: negative target, no positive train
    history, then target outside the positive training catalogue.
    """
    threshold = float(params.get("positive_threshold", 4.0))
    positive = train.filter(F.col("rating") >= threshold)
    users = positive.select("user_idx").distinct().withColumn("known_user", F.lit(True))
    items = positive.select("item_idx").distinct().withColumn("known_item", F.lit(True))

    def eligible(split):
        annotated = (
            split.join(users, "user_idx", "left")
            .join(items, "item_idx", "left")
            .withColumn(
                "exclusion",
                F.when(F.col("rating") < threshold, "negative")
                .when(F.col("known_user").isNull(), "no_positive_history")
                .when(F.col("known_item").isNull(), "unknown_item")
                .otherwise("eligible"),
            )
        )
        counts = {
            r["exclusion"]: r["count"]
            for r in annotated.groupBy("exclusion").count().collect()
        }
        stats = {
            key: int(counts.get(key, 0))
            for key in ("eligible", "negative", "no_positive_history", "unknown_item")
        }
        stats["total"] = sum(stats.values())
        return annotated.filter(F.col("exclusion") == "eligible").select(
            *split.columns
        ), stats

    valid_targets, valid_stats = eligible(validation)
    test_targets, test_stats = eligible(test)
    seen = train.unionByName(validation).select("user_idx", "item_idx").distinct()
    return (
        positive,
        valid_targets,
        test_targets,
        seen,
        {
            "positive_threshold": threshold,
            "validation": valid_stats,
            "test": test_stats,
            "catalog_size": items.count(),
        },
    )
