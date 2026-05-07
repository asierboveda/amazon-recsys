"""Tests for SASRec pipeline nodes."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

from amazon_recsys.pipelines.recommender_sasrec.nodes import (
    build_sasrec_sequences,
    evaluate_sasrec_ranking_metrics,
)

_PYTHON_BIN = sys.executable


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = _PYTHON_BIN
    os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON_BIN
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-recommender-sasrec")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executorEnv.PYTHONPATH", os.pathsep.join(sys.path))
        .getOrCreate()
    )
    yield session
    session.stop()


def test_build_sasrec_sequences_orders_history_and_offsets_items(spark):
    train = spark.createDataFrame(
        [
            (0, 20, 4.0, 2000),
            (0, 10, 5.0, 1000),
            (0, 30, 4.0, 3000),
            (1, 40, 5.0, 1000),
        ],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )
    validation = spark.createDataFrame(
        [(0, 50, 4.0, 4000)],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )
    test = spark.createDataFrame(
        [(0, 60, 5.0, 5000)],
        schema=["user_idx", "item_idx", "rating", "timestamp"],
    )

    train_seq, validation_seq, test_seq, seen_items, stats = build_sasrec_sequences(
        train,
        validation,
        test,
        {"max_seq_len": 2, "min_sequence_length": 1, "item_id_offset": 1},
    )

    train_rows = {
        row["user_idx"]: row.asDict()
        for row in train_seq.collect()
    }
    validation_row = validation_seq.collect()[0].asDict()
    test_row = test_seq.collect()[0].asDict()

    assert train_rows[0]["sequence"] == [21, 31]
    assert train_rows[0]["target_item"] == 31
    assert train_rows[1]["sequence"] == [41]
    assert validation_row["sequence"] == [21, 31]
    assert validation_row["target_item"] == 51
    assert test_row["sequence"] == [31, 51]
    assert test_row["target_item"] == 61
    assert seen_items.where("user_idx = 0").count() == 4
    assert stats["max_seq_len"] == 2
    assert stats["item_id_offset"] == 1


def test_evaluate_sasrec_ranking_metrics_uses_target_item(spark):
    recs = spark.createDataFrame(
        [
            (0, 60, 0.9, 1),
            (0, 70, 0.8, 2),
            (1, 80, 0.7, 1),
        ],
        schema=["user_idx", "item_idx", "score", "rank"],
    )
    test_sequences = spark.createDataFrame(
        [
            (0, [10, 20], 60),
            (1, [30, 40], 90),
        ],
        schema=["user_idx", "sequence", "target_item"],
    )

    metrics = evaluate_sasrec_ranking_metrics(
        recs, test_sequences, {"k_values": [1, 2], "item_id_offset": 0}
    )

    assert metrics == {"recall@1": 0.5, "recall@2": 0.5}
