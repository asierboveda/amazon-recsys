"""Tests for LightGCN pipeline nodes."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

from amazon_recsys.pipelines.recommender_lightgcn.nodes import (
    evaluate_lightgcn_ranking_metrics,
    prepare_lightgcn_interactions,
)

_PYTHON_BIN = sys.executable


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = _PYTHON_BIN
    os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_BIN
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-recommender-lightgcn")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executorEnv.PYTHONPATH", os.pathsep.join(sys.path))
        .getOrCreate()
    )
    yield session
    session.stop()


def test_prepare_lightgcn_interactions_filters_positive_ratings(spark):
    train = spark.createDataFrame(
        [
            (0, 10, 5.0, 1000),
            (0, 20, 3.0, 1001),
            (1, 30, 4.0, 1002),
        ],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )
    validation = spark.createDataFrame(
        [(0, 40, 2.0, 1003)],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )
    test = spark.createDataFrame(
        [(1, 50, 5.0, 1004)],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )

    train_pos, validation_pos, test_pos, stats = prepare_lightgcn_interactions(
        train,
        validation,
        test,
        {"positive_threshold": 4.0},
    )

    assert sorted(train_pos.select("item_idx").toPandas()["item_idx"].tolist()) == [
        10,
        30,
    ]
    assert validation_pos.count() == 0
    assert test_pos.count() == 1
    assert stats["num_train_interactions"] == 2
    assert stats["positive_threshold"] == 4.0


def test_evaluate_lightgcn_ranking_metrics_reuses_recall_at_k(spark):
    recs = spark.createDataFrame(
        [
            (0, 10, 0.9, 1),
            (0, 20, 0.8, 2),
            (1, 30, 0.7, 1),
        ],
        schema=["user_idx", "item_idx", "score", "rank"],
    )
    test = spark.createDataFrame(
        [
            (0, 10, 5.0, 1000),
            (0, 99, 5.0, 1001),
            (1, 30, 4.0, 1002),
        ],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )

    metrics = evaluate_lightgcn_ranking_metrics(recs, test, {"k_values": [1, 2]})

    assert metrics == {"recall@1": 0.75, "recall@2": 0.75}
