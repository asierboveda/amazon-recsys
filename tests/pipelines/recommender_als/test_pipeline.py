"""Tests for the recommender_als pipeline nodes."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

from amazon_recsys.pipelines.recommender_als.nodes import (
    _compute_recall_at_k,
    compare_model_metrics,
    evaluate_popularity_baseline,
)

_PYTHON_BIN = sys.executable


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = _PYTHON_BIN
    os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_BIN
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-recommender-als")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executorEnv.PYTHONPATH", os.pathsep.join(sys.path))
        .getOrCreate()
    )
    yield session
    session.stop()


class TestComputeRecallAtK:
    def test_perfect_recall(self, spark):
        recs = spark.createDataFrame(
            [
                (0, 10, 1),
                (0, 20, 2),
                (0, 30, 3),
            ],
            schema=["user_idx", "item_idx", "rank"],
        )
        gt = spark.createDataFrame(
            [(0, 10), (0, 20)],
            schema=["user_idx", "item_idx"],
        )
        recall = _compute_recall_at_k(recs, gt, k=2)
        assert recall == 1.0

    def test_partial_recall(self, spark):
        recs = spark.createDataFrame(
            [
                (0, 10, 1),
                (0, 40, 2),
            ],
            schema=["user_idx", "item_idx", "rank"],
        )
        gt = spark.createDataFrame(
            [(0, 10), (0, 20)],
            schema=["user_idx", "item_idx"],
        )
        recall = _compute_recall_at_k(recs, gt, k=2)
        assert recall == 0.5

    def test_zero_recall(self, spark):
        recs = spark.createDataFrame(
            [(0, 99, 1)],
            schema=["user_idx", "item_idx", "rank"],
        )
        gt = spark.createDataFrame(
            [(0, 10)],
            schema=["user_idx", "item_idx"],
        )
        recall = _compute_recall_at_k(recs, gt, k=1)
        assert recall == 0.0

    def test_respects_k_cutoff(self, spark):
        recs = spark.createDataFrame(
            [
                (0, 10, 1),
                (0, 20, 2),
                (0, 30, 3),
            ],
            schema=["user_idx", "item_idx", "rank"],
        )
        gt = spark.createDataFrame(
            [(0, 30)],
            schema=["user_idx", "item_idx"],
        )
        recall_k1 = _compute_recall_at_k(recs, gt, k=1)
        recall_k3 = _compute_recall_at_k(recs, gt, k=3)
        assert recall_k1 == 0.0
        assert recall_k3 == 1.0


class TestPopularityBaseline:
    def test_evaluate_popularity_baseline(self, spark):
        recs = spark.createDataFrame(
            [
                (0, 10, 150.0, 1),
                (0, 20, 100.0, 2),
            ],
            schema=["user_idx", "item_idx", "score", "rank"],
        )
        als_test = spark.createDataFrame(
            [
                (0, 10, 4.0, 1000, "u1", "p1"),
                (0, 20, 3.0, 2000, "u1", "p2"),
            ],
            schema=["user_idx", "item_idx", "rating", "timestamp", "user_id", "item_id"],
        )
        metrics = evaluate_popularity_baseline(
            recs, als_test, {"k_values": [1, 2]}
        )
        assert metrics["recall@1"] == 0.5
        assert metrics["recall@2"] == 1.0


class TestCompareModelMetrics:
    def test_als_beats_popularity(self):
        als_reg = {"rmse": 0.8}
        als_rank = {"recall@10": 0.15, "recall@20": 0.22}
        pop_rank = {"recall@10": 0.10, "recall@20": 0.18}
        result = compare_model_metrics(als_reg, als_rank, pop_rank)
        assert result["als_beats_popularity"] is True

    def test_als_does_not_beat_popularity(self):
        als_reg = {"rmse": 1.2}
        als_rank = {"recall@10": 0.08, "recall@20": 0.12}
        pop_rank = {"recall@10": 0.10, "recall@20": 0.18}
        result = compare_model_metrics(als_reg, als_rank, pop_rank)
        assert result["als_beats_popularity"] is False

    def test_comparison_has_all_keys(self):
        als_reg = {"rmse": 0.8}
        als_rank = {"recall@10": 0.15}
        pop_rank = {"recall@10": 0.10}
        result = compare_model_metrics(als_reg, als_rank, pop_rank)
        assert "als_regression" in result
        assert "als_ranking" in result
        assert "popularity_ranking" in result
        assert "als_beats_popularity" in result
