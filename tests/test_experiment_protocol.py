"""Regression tests for the shared warm-start experiment."""

import pytest
from pyspark.sql import SparkSession

from amazon_recsys.experiment import prepare_experiment


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_common_population_requires_positive_history_and_known_target(spark):
    schema = "user_idx int, item_idx int, rating float, timestamp long"
    train = spark.createDataFrame(
        [(0, 0, 5.0, 1), (1, 1, 2.0, 1), (2, 2, 4.0, 1), (3, 3, 5.0, 1)], schema
    )
    validation = spark.createDataFrame(
        [(0, 2, 4.0, 2), (1, 0, 4.0, 2), (2, 0, 1.0, 2), (3, 2, 4.0, 2)], schema
    )
    test = spark.createDataFrame(
        [(0, 2, 5.0, 3), (1, 0, 5.0, 3), (2, 9, 5.0, 3), (3, 0, 3.0, 3)], schema
    )
    positive, val, eligible, seen, stats = prepare_experiment(
        train, validation, test, {"positive_threshold": 4}
    )
    assert {(r.user_idx, r.item_idx) for r in positive.collect()} == {
        (0, 0),
        (2, 2),
        (3, 3),
    }
    assert {(r.user_idx, r.item_idx) for r in eligible.collect()} == {(0, 2)}
    assert {(r.user_idx, r.item_idx) for r in val.collect()} == {(0, 2), (3, 2)}
    assert (1, 1) in {(r.user_idx, r.item_idx) for r in seen.collect()}
    assert stats["test"]["total"] == 4
    assert stats["test"]["eligible"] == 1
    assert stats["test"]["negative"] == 1
    assert stats["test"]["no_positive_history"] == 1
    assert stats["test"]["unknown_item"] == 1


def test_models_train_select_validation_and_rank_same_users(
    spark, tmp_path, monkeypatch
):
    from amazon_recsys.pipelines.recommender_als.nodes import (
        build_popularity_baseline,
        generate_als_recommendations,
        train_als_implicit_model,
    )
    from amazon_recsys.pipelines.recommender_lightgcn.nodes import (
        generate_lightgcn_recommendations,
        train_lightgcn_model,
    )
    from amazon_recsys.pipelines.recommender_sasrec.nodes import (
        build_sasrec_sequences,
        generate_sasrec_recommendations,
        train_sasrec_model,
    )

    import copy
    import torch
    from amazon_recsys.pipelines.recommender_lightgcn.model import LightGCN
    from amazon_recsys.pipelines.recommender_sasrec.model import SASRecModel

    captured = {"lightgcn": [], "sasrec": []}
    for cls, name in ((LightGCN, "lightgcn"), (SASRecModel, "sasrec")):
        original = cls.state_dict

        def record(self, original=original, name=name):
            state = original(self)
            captured[name].append(copy.deepcopy(state))
            return state

        monkeypatch.setattr(cls, "state_dict", record)

    schema = "user_idx int, item_idx int, rating float, timestamp long"
    train = spark.createDataFrame(
        [(u, 2 * ((u + j) % 6), 5.0, j) for u in range(6) for j in range(2)], schema
    )
    val = spark.createDataFrame(
        [(u, 2 * ((u + 2) % 6), 5.0, 2) for u in range(6)], schema
    )
    test = spark.createDataFrame(
        [(u, 2 * ((u + 3) % 6), 5.0, 3) for u in range(6)], schema
    )
    positive, val_gt, test_gt, seen, stats = prepare_experiment(train, val, test, {})
    assert stats["test"]["eligible"] == 6
    common = {"k": 3}
    pop = build_popularity_baseline(positive, test_gt, common, seen)
    als = train_als_implicit_model(
        positive,
        {
            "rank": 2,
            "maxIter": 1,
            "implicitPrefs": True,
            "model_path": str(tmp_path / "als"),
        },
    )
    als_recs = generate_als_recommendations(als, positive, test_gt, common, seen)
    lg = train_lightgcn_model(
        positive,
        val_gt,
        {
            "embedding_dim": 4,
            "n_layers": 1,
            "batch_size": 4,
            "epochs": 2,
            "model_path": str(tmp_path / "lg.pt"),
        },
        train,
    )
    assert lg["best_epoch"] == 1
    lg_saved = torch.load(lg["model_path"], map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    for group, weights in captured["lightgcn"][0].items():
        for key, value in weights.items():
            assert torch.equal(value.cpu(), lg_saved[group][key])
    lg_recs = generate_lightgcn_recommendations(lg, positive, test_gt, common, seen)
    from amazon_recsys.recommender_metrics import compute_recall_at_k

    lg_val_recs = generate_lightgcn_recommendations(
        lg, positive, val_gt, {"k": 20}, train
    )
    assert compute_recall_at_k(lg_val_recs, val_gt, 20) == pytest.approx(
        lg["best_validation_recall_at_20"]
    )
    sas_train, sas_val, sas_test, sas_seen, _ = build_sasrec_sequences(
        train, val, test, {"max_seq_len": 4}, val_gt, test_gt
    )
    sas = train_sasrec_model(
        sas_train,
        sas_val,
        {
            "hidden_size": 4,
            "inner_size": 8,
            "n_heads": 2,
            "n_layers": 1,
            "max_seq_len": 4,
            "batch_size": 4,
            "epochs": 2,
            "model_path": str(tmp_path / "sas.pt"),
        },
        train,
    )
    assert sas["best_epoch"] == 1
    sas_saved = torch.load(sas["model_path"], map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    for key, value in captured["sasrec"][0].items():
        assert torch.equal(value.cpu(), sas_saved[key])
    sas_recs = generate_sasrec_recommendations(sas, sas_test, sas_seen, common)
    sas_val_recs = generate_sasrec_recommendations(sas, sas_val, train, {"k": 20})
    assert compute_recall_at_k(sas_val_recs, val_gt, 20) == pytest.approx(
        sas["best_validation_recall_at_20"]
    )
    known = {(r.user_idx, r.item_idx) for r in seen.collect()}
    for recs in (pop, als_recs, lg_recs, sas_recs):
        rows = recs.collect()
        assert {r.user_idx for r in rows} == set(range(6))
        assert all((r.user_idx, r.item_idx) not in known for r in rows)
        assert all(r.item_idx in {0, 2, 4, 6, 8, 10} for r in rows)
