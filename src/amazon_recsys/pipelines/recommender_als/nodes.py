"""Recommender-ALS pipeline nodes.

All nodes use PySpark exclusively. No Pandas, no collect() on large data.
The trained ALS model is persisted to disk via Spark's native save/load
because Kedro's pickle-based serialisation cannot round-trip ALSModel.
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def train_als_explicit_model(
    train: DataFrame, als_params: dict
) -> dict:
    """Train a Spark ALS model for explicit-feedback collaborative filtering.

    The trained model is persisted to disk using model.write().overwrite().save()
    and only a lightweight metadata dictionary is returned through the Kedro
    catalog (pickled as a dict).

    Args:
        train:        Spark DataFrame with user_idx, item_idx, rating.
        als_params:   Dictionary of ALS hyper-parameters.

    Returns:
        A dictionary with model metadata and the path where the model was saved.
    """
    spark = SparkSession.builder.getOrCreate()

    rank = als_params.get("rank", 50)
    max_iter = als_params.get("maxIter", 10)
    reg_param = als_params.get("regParam", 0.1)
    nonnegative = als_params.get("nonnegative", True)
    implicit_prefs = als_params.get("implicitPrefs", False)
    cold_start_strategy = als_params.get("coldStartStrategy", "drop")
    seed = als_params.get("seed", 42)
    model_path = als_params.get("model_path", "data/06_models/als_model_spark")

    als = ALS(
        userCol=als_params.get("user_col", "user_idx"),
        itemCol=als_params.get("item_col", "item_idx"),
        ratingCol=als_params.get("rating_col", "rating"),
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        nonnegative=nonnegative,
        implicitPrefs=implicit_prefs,
        coldStartStrategy=cold_start_strategy,
        seed=seed,
    )

    model = als.fit(train)

    model.write().overwrite().save(model_path)

    return {
        "model_path": model_path,
        "rank": rank,
        "maxIter": max_iter,
        "regParam": reg_param,
        "nonnegative": nonnegative,
        "implicitPrefs": implicit_prefs,
        "coldStartStrategy": cold_start_strategy,
        "seed": seed,
    }


def _load_model(model_info: dict) -> ALSModel:
    """Load a previously saved ALSModel from its metadata dictionary."""
    return ALSModel.load(model_info["model_path"])


def generate_als_recommendations(
    model_info: dict, als_test: DataFrame, recommender_params: dict
) -> DataFrame:
    """Generate top-K recommendations for every user present in the test set.

    Uses recommendForUserSubset so only test users receive recommendations.

    Args:
        model_info:          Dictionary returned by train_als_explicit_model.
        als_test:            Test-set interactions (at least user_idx column).
        recommender_params:  Dict with `k` (int) — number of recs per user.

    Returns:
        DataFrame with columns [user_idx, item_idx, score, rank].
    """
    k = recommender_params.get("k", 20)
    model = _load_model(model_info)

    users = als_test.select("user_idx").distinct()

    raw_recs = model.recommendForUserSubset(users, k)

    recs = raw_recs.withColumn("rec", F.explode("recommendations"))
    recs = recs.select(
        F.col("user_idx"),
        F.col("rec.item_idx").cast("int").alias("item_idx"),
        F.col("rec.rating").alias("score"),
    )

    window = Window.partitionBy("user_idx").orderBy(F.col("score").desc())
    recs = recs.withColumn("rank", F.row_number().over(window))

    return recs


def evaluate_rmse(model_info: dict, als_validation: DataFrame) -> dict:
    """Compute RMSE of the ALS model on a hold-out validation set.

    Args:
        model_info:      Dictionary returned by train_als_explicit_model.
        als_validation:  Validation interactions (user_idx, item_idx, rating).

    Returns:
        Dictionary ``{"rmse": <float_value>}``.
    """
    model = _load_model(model_info)

    predictions = model.transform(als_validation)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    rmse = evaluator.evaluate(predictions)

    return {"rmse": round(rmse, 6)}


def _compute_recall_at_k(
    recommendations: DataFrame, ground_truth: DataFrame, k: int
) -> float:
    """Compute Recall@k averaged across users.

    Args:
        recommendations: DataFrame with [user_idx, item_idx, rank].
        ground_truth:    DataFrame with [user_idx, item_idx] (relevant items).
        k:               Cut-off rank.

    Returns:
        Average Recall@k (float).
    """
    relevant = ground_truth.select("user_idx", "item_idx").distinct()

    rel_count = relevant.groupBy("user_idx").agg(
        F.count("item_idx").alias("total_relevant")
    )

    top_k = recommendations.filter(F.col("rank") <= k)

    hits = top_k.join(relevant, ["user_idx", "item_idx"], "inner")

    hit_count = hits.groupBy("user_idx").agg(
        F.count("item_idx").alias("num_hits")
    )

    user_recall = rel_count.join(hit_count, "user_idx", "left").fillna(0)
    user_recall = user_recall.withColumn(
        "recall", F.col("num_hits") / F.col("total_relevant")
    )

    avg_recall = user_recall.select(F.avg("recall")).first()[0]
    return float(avg_recall) if avg_recall is not None else 0.0


def evaluate_ranking_metrics(
    als_recommendations_top_k: DataFrame,
    als_test: DataFrame,
    evaluation_params: dict,
) -> dict:
    """Compute ranking metrics (Recall@K) for ALS recommendations.

    Args:
        als_recommendations_top_k:  ALS recommendations.
        als_test:                   Test interactions (ground truth).
        evaluation_params:          Dict with `k_values` (list of int).

    Returns:
        Dictionary mapping "recall@<k>" to the computed value.
    """
    k_values = evaluation_params.get("k_values", [10, 20])

    metrics = {}
    for k in k_values:
        recall = _compute_recall_at_k(als_recommendations_top_k, als_test, k)
        metrics[f"recall@{k}"] = round(recall, 6)

    return metrics


def build_popularity_baseline(
    als_train: DataFrame,
    als_test: DataFrame,
    recommender_params: dict,
) -> DataFrame:
    """Build a non-personalised popularity baseline.

    Items are scored by interaction count (tie-broken by average rating).
    For each test user, the top-K globally popular items are recommended,
    excluding items already seen in the training set.

    Args:
        als_train:          Training interactions.
        als_test:           Test interactions (used to extract user_idx set).
        recommender_params: Dict with `k` (int).

    Returns:
        DataFrame with [user_idx, item_idx, score, rank].
    """
    k = recommender_params.get("k", 20)

    popularity = als_train.groupBy("item_idx").agg(
        F.count("*").alias("score"),
        F.avg("rating").alias("avg_rating"),
    )

    window_global = Window.orderBy(
        F.col("score").desc(), F.col("avg_rating").desc()
    )
    popularity = popularity.withColumn(
        "_glob", F.row_number().over(window_global)
    )

    num_candidates = max(k * 10, 200)
    top_popular = popularity.filter(F.col("_glob") <= num_candidates).select(
        "item_idx", "score"
    )

    test_users = als_test.select("user_idx").distinct()
    seen_items = als_train.select("user_idx", "item_idx").distinct()

    recs = test_users.crossJoin(top_popular)
    recs = recs.join(seen_items, ["user_idx", "item_idx"], "left_anti")

    window_user = Window.partitionBy("user_idx").orderBy(
        F.col("score").desc()
    )
    recs = recs.withColumn("rank", F.row_number().over(window_user))
    recs = recs.filter(F.col("rank") <= k)

    return recs


def evaluate_popularity_baseline(
    popularity_recommendations_top_k: DataFrame,
    als_test: DataFrame,
    evaluation_params: dict,
) -> dict:
    """Compute Recall@K for the popularity baseline.

    Args:
        popularity_recommendations_top_k: Popularity recommendations.
        als_test:                         Test interactions (ground truth).
        evaluation_params:               Dict with `k_values` (list of int).

    Returns:
        Dictionary mapping "recall@<k>" to the computed value.
    """
    k_values = evaluation_params.get("k_values", [10, 20])

    metrics = {}
    for k in k_values:
        recall = _compute_recall_at_k(
            popularity_recommendations_top_k, als_test, k
        )
        metrics[f"recall@{k}"] = round(recall, 6)

    return metrics


def compare_model_metrics(
    als_regression_metrics: dict,
    als_ranking_metrics: dict,
    popularity_ranking_metrics: dict,
) -> dict:
    """Build a comparison report between ALS and the popularity baseline.

    Args:
        als_regression_metrics:    RMSE dict from evaluate_rmse.
        als_ranking_metrics:       Recall@K dict from evaluate_ranking_metrics.
        popularity_ranking_metrics: Recall@K dict from evaluate_popularity_baseline.

    Returns:
        A comparison dictionary.
    """
    als_beats_pop = True
    for k, val in als_ranking_metrics.items():
        pop_val = popularity_ranking_metrics.get(k, -1)
        if val <= pop_val:
            als_beats_pop = False
            break

    return {
        "als_regression": als_regression_metrics,
        "als_ranking": als_ranking_metrics,
        "popularity_ranking": popularity_ranking_metrics,
        "als_beats_popularity": als_beats_pop,
    }
