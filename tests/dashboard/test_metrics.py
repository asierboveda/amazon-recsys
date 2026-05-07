"""Tests for dashboard metric helpers."""

import pandas as pd

from amazon_recsys.dashboard.metrics import (
    average_jaccard_at_k,
    catalog_coverage_at_k,
    ranking_metrics_at_k,
)


def test_ranking_metrics_at_k_computes_recall_precision_and_ndcg():
    recommendations = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 10, "rank": 1},
            {"user_idx": 1, "item_idx": 99, "rank": 2},
            {"user_idx": 2, "item_idx": 30, "rank": 1},
            {"user_idx": 2, "item_idx": 40, "rank": 2},
        ]
    )
    ground_truth = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 10},
            {"user_idx": 1, "item_idx": 20},
            {"user_idx": 2, "item_idx": 50},
        ]
    )

    result = ranking_metrics_at_k(recommendations, ground_truth, k=2)

    assert result["recall"] == 0.25
    assert result["precision"] == 0.25
    assert round(result["ndcg"], 6) == 0.306574


def test_catalog_coverage_at_k_counts_distinct_recommended_items():
    recommendations = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 10, "rank": 1},
            {"user_idx": 1, "item_idx": 20, "rank": 2},
            {"user_idx": 2, "item_idx": 20, "rank": 1},
        ]
    )

    assert catalog_coverage_at_k(recommendations, catalog_size=10, k=2) == 0.2


def test_average_jaccard_at_k_compares_user_recommendation_sets():
    left = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 10, "rank": 1},
            {"user_idx": 1, "item_idx": 20, "rank": 2},
            {"user_idx": 2, "item_idx": 30, "rank": 1},
        ]
    )
    right = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 20, "rank": 1},
            {"user_idx": 1, "item_idx": 30, "rank": 2},
            {"user_idx": 2, "item_idx": 30, "rank": 1},
        ]
    )

    assert round(average_jaccard_at_k(left, right, k=2), 6) == 0.666667
