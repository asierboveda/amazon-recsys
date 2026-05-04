"""Tests for the data_processing pipeline nodes."""

import os
import sys

import pytest
from pyspark.sql import SparkSession, functions as F

from amazon_recsys.pipelines.data_processing.nodes import (
    clean_recommender_interactions,
    deduplicate_user_item_interactions,
    filter_min_activity,
    select_recommender_columns,
    temporal_train_validation_test_split,
)

_PYTHON_BIN = sys.executable


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = _PYTHON_BIN
    os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_BIN
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-data-processing")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executorEnv.PYTHONPATH", os.pathsep.join(sys.path))
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_raw(spark):
    return spark.createDataFrame(
        [
            ("u1", "p1", 4.0, 1000, True, 0),
            ("u1", "p2", 5.0, 2000, False, 1),
            ("u2", "p1", 3.0, 1500, True, -1),
            ("u2", "p2", None, 2500, True, 0),
            (None, "p3", 4.0, 3000, True, 0),
            ("u3", None, 2.0, 4000, True, 0),
            ("u3", "p1", 6.0, 5000, True, 0),
            ("u4", "p3", 3.0, 3500, True, 0),
        ],
        schema=[
            "user_id", "parent_asin", "rating", "timestamp",
            "verified_purchase", "helpful_vote",
        ],
    )


class TestSelectRecommenderColumns:
    def test_renames_parent_asin_to_item_id(self, sample_raw):
        result = select_recommender_columns(sample_raw)
        assert "item_id" in result.columns
        assert "parent_asin" not in result.columns
        assert "user_id" in result.columns
        assert "rating" in result.columns

    def test_keeps_expected_columns(self, sample_raw):
        result = select_recommender_columns(sample_raw)
        expected = {"user_id", "item_id", "rating", "timestamp",
                    "verified_purchase", "helpful_vote"}
        assert set(result.columns) == expected

    def test_does_not_drop_rows(self, sample_raw):
        result = select_recommender_columns(sample_raw)
        assert result.count() == sample_raw.count()


class TestCleanRecommenderInteractions:
    def test_removes_null_user_item_rating(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 4.0, 1000, True, 0),
                (None, "p1", 4.0, 1000, True, 0),
                ("u1", None, 4.0, 1000, True, 0),
                ("u1", "p1", None, 1000, True, 0),
            ],
            schema=["user_id", "item_id", "rating", "timestamp",
                    "verified_purchase", "helpful_vote"],
        )
        result = clean_recommender_interactions(df)
        assert result.count() == 1

    def test_filters_ratings_out_of_range(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 0.0, 1000, True, 0),
                ("u1", "p2", 5.0, 1000, True, 0),
                ("u1", "p3", 6.0, 1000, True, 0),
            ],
            schema=["user_id", "item_id", "rating", "timestamp",
                    "verified_purchase", "helpful_vote"],
        )
        result = clean_recommender_interactions(df)
        assert result.count() == 1

    def test_replaces_negative_helpful_vote(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 4.0, 1000, True, -5),
                ("u1", "p2", 5.0, 1000, True, 3),
            ],
            schema=["user_id", "item_id", "rating", "timestamp",
                    "verified_purchase", "helpful_vote"],
        )
        result = clean_recommender_interactions(df)
        helpful_votes = [r.helpful_vote for r in result.collect()]
        assert all(v >= 0 for v in helpful_votes)


class TestDeduplicateUserItemInteractions:
    def test_keeps_most_recent_per_pair(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 4.0, 1000, True, 0),
                ("u1", "p1", 5.0, 2000, True, 0),
                ("u2", "p1", 3.0, 1500, True, 0),
            ],
            schema=["user_id", "item_id", "rating", "timestamp",
                    "verified_purchase", "helpful_vote"],
        )
        result = deduplicate_user_item_interactions(df)
        assert result.count() == 2
        for row in result.filter("user_id = 'u1' AND item_id = 'p1'").collect():
            assert row.rating == 5.0
            assert row.timestamp == 2000


class TestFilterMinActivity:
    def test_filters_users_and_items(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 4.0, 1000),
                ("u1", "p2", 3.0, 2000),
                ("u2", "p1", 5.0, 1500),
            ],
            schema=["user_id", "item_id", "rating", "timestamp"],
        )
        result = filter_min_activity(df, 2, 1)
        user_ids = {r.user_id for r in result.select("user_id").collect()}
        assert "u2" not in user_ids
        assert "u1" in user_ids

    def test_filters_low_activity_items(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", "p1", 4.0, 1000),
                ("u1", "p2", 3.0, 2000),
                ("u2", "p1", 5.0, 1500),
                ("u3", "p2", 2.0, 1600),
            ],
            schema=["user_id", "item_id", "rating", "timestamp"],
        )
        result = filter_min_activity(df, 1, 2)
        item_ids = {r.item_id for r in result.select("item_id").collect()}
        assert "p1" in item_ids
        assert "p2" in item_ids


class TestTemporalSplit:
    def test_uses_latest_for_test(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", 0, 0, 4.0, 3000),
                ("u1", 0, 1, 3.0, 1000),
                ("u1", 0, 2, 5.0, 2000),
            ],
            schema=["user_id", "user_idx", "item_idx", "rating", "timestamp"],
        )
        best = df.orderBy("timestamp", ascending=False).first()
        train, val, test = temporal_train_validation_test_split(
            df, {"use_validation": False, "timestamp_col": "timestamp"}
        )
        test_rows = test.collect()
        assert len(test_rows) == 1
        r = test_rows[0]
        assert r.timestamp == best.timestamp

    def test_uses_second_latest_for_validation(self, spark):
        df = spark.createDataFrame(
            [
                ("u1", 0, 0, 4.0, 3000),
                ("u1", 0, 1, 3.0, 2000),
                ("u1", 0, 2, 5.0, 1000),
            ],
            schema=["user_id", "user_idx", "item_idx", "rating", "timestamp"],
        )
        train, val, test = temporal_train_validation_test_split(
            df, {"use_validation": True, "timestamp_col": "timestamp"}
        )
        assert val.count() == 1
        assert train.count() == 1
        assert test.count() == 1
