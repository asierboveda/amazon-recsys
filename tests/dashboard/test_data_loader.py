"""Tests for dashboard data preparation helpers."""

import pandas as pd

from amazon_recsys.dashboard.data_loader import (
    available_recommendation_tables,
    model_availability,
    top_recommendations_with_hits,
)


class DummyCatalog:
    def __init__(self, datasets):
        self.datasets = datasets

    def load(self, dataset_name):
        if dataset_name not in self.datasets:
            raise FileNotFoundError(dataset_name)
        return self.datasets[dataset_name]


def test_model_availability_marks_missing_datasets():
    catalog = DummyCatalog({"als_recommendations_top_k": pd.DataFrame()})
    mapping = {
        "ALS": "als_recommendations_top_k",
        "SASRec": "sasrec_recommendations_top_k",
    }

    result = model_availability(catalog, mapping)

    assert result.loc[result["model"] == "ALS", "available"].iloc[0] is True
    assert result.loc[result["model"] == "SASRec", "available"].iloc[0] is False


def test_available_recommendation_tables_loads_only_existing_models():
    catalog = DummyCatalog(
        {
            "als_recommendations_top_k": pd.DataFrame(
                [{"user_idx": 1, "item_idx": 10, "score": 0.8, "rank": 1}]
            )
        }
    )
    mapping = {
        "ALS": "als_recommendations_top_k",
        "SASRec": "sasrec_recommendations_top_k",
    }

    result = available_recommendation_tables(catalog, mapping)

    assert list(result) == ["ALS"]
    assert result["ALS"].iloc[0]["item_idx"] == 10


def test_top_recommendations_with_hits_flags_ground_truth_items():
    recommendations = {
        "ALS": pd.DataFrame(
            [
                {"user_idx": 1, "item_idx": 10, "score": 0.8, "rank": 1},
                {"user_idx": 1, "item_idx": 99, "score": 0.7, "rank": 2},
            ]
        )
    }
    ground_truth = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 10},
            {"user_idx": 1, "item_idx": 20},
        ]
    )

    result = top_recommendations_with_hits(recommendations, ground_truth, 1, k=2)

    assert result.loc[result["item_idx"] == 10, "hit"].iloc[0] is True
    assert result.loc[result["item_idx"] == 99, "hit"].iloc[0] is False
