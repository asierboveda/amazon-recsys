"""Small Pandas metric helpers for the presentation dashboard."""

from __future__ import annotations

import math

import pandas as pd


def _top_k(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations.copy()
    return recommendations[recommendations["rank"] <= k].copy()


def ranking_metrics_at_k(
    recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int
) -> dict[str, float]:
    """Compute Recall, Precision and NDCG averaged by user."""
    if ground_truth.empty:
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}

    top_k = _top_k(recommendations, k)
    users = sorted(ground_truth["user_idx"].unique())
    recalls = []
    precisions = []
    ndcgs = []

    for user_idx in users:
        relevant = set(
            ground_truth.loc[ground_truth["user_idx"] == user_idx, "item_idx"]
        )
        user_recs = top_k[top_k["user_idx"] == user_idx].sort_values("rank")
        recommended = list(user_recs["item_idx"])
        hits = [item_idx for item_idx in recommended if item_idx in relevant]

        recalls.append(len(hits) / len(relevant) if relevant else 0.0)
        precisions.append(len(hits) / len(recommended) if recommended else 0.0)

        dcg = 0.0
        for rank, item_idx in enumerate(recommended, start=1):
            if item_idx in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "recall": round(sum(recalls) / len(recalls), 6),
        "precision": round(sum(precisions) / len(precisions), 6),
        "ndcg": round(sum(ndcgs) / len(ndcgs), 6),
    }


def catalog_coverage_at_k(
    recommendations: pd.DataFrame, catalog_size: int, k: int
) -> float:
    """Compute distinct recommended items divided by catalog size."""
    if catalog_size <= 0 or recommendations.empty:
        return 0.0
    distinct_items = _top_k(recommendations, k)["item_idx"].nunique()
    return round(distinct_items / catalog_size, 6)


def average_jaccard_at_k(
    left: pd.DataFrame, right: pd.DataFrame, k: int
) -> float:
    """Compute average user-level Jaccard overlap between two top-K lists."""
    users = sorted(set(left["user_idx"]).union(set(right["user_idx"])))
    if not users:
        return 0.0

    left_top = _top_k(left, k)
    right_top = _top_k(right, k)
    values = []
    for user_idx in users:
        left_items = set(left_top.loc[left_top["user_idx"] == user_idx, "item_idx"])
        right_items = set(
            right_top.loc[right_top["user_idx"] == user_idx, "item_idx"]
        )
        union = left_items | right_items
        if not union:
            values.append(0.0)
        else:
            values.append(len(left_items & right_items) / len(union))
    return round(sum(values) / len(values), 6)


def build_metric_table(
    recommendations_by_model: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    catalog_size: int,
    k_values: list[int],
) -> pd.DataFrame:
    """Build a long model-metric table for dashboard charts."""
    rows = []
    for model_name, recs in recommendations_by_model.items():
        for k in k_values:
            ranking = ranking_metrics_at_k(recs, ground_truth, k)
            rows.append(
                {
                    "model": model_name,
                    "k": k,
                    **ranking,
                    "coverage": catalog_coverage_at_k(recs, catalog_size, k),
                }
            )
    return pd.DataFrame(rows)
