"""Data loading helpers for the Dash dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MODEL_RECOMMENDATION_DATASETS = {
    "Popularity": "popularity_recommendations_top_k",
    "ALS": "als_recommendations_top_k",
    "LightGCN": "lightgcn_recommendations_top_k",
    "SASRec": "sasrec_recommendations_top_k",
}


def _to_pandas(data: Any, limit: int | None = None) -> pd.DataFrame:
    """Convert Spark/Pandas-like datasets to Pandas after optional limiting."""
    if isinstance(data, pd.DataFrame):
        return data.head(limit).copy() if limit is not None else data.copy()
    if hasattr(data, "limit") and limit is not None:
        data = data.limit(limit)
    if hasattr(data, "toPandas"):
        return data.toPandas()
    return pd.DataFrame(data)


def _safe_load(catalog: Any, dataset_name: str) -> Any | None:
    try:
        return catalog.load(dataset_name)
    except Exception:
        return None


def model_availability(
    catalog: Any,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return availability status for recommendation datasets."""
    mapping = mapping or MODEL_RECOMMENDATION_DATASETS
    rows = []
    for model_name, dataset_name in mapping.items():
        data = _safe_load(catalog, dataset_name)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "available": data is not None,
            }
        )
    result = pd.DataFrame(rows)
    if "available" in result:
        result["available"] = result["available"].astype(object)
    return result


def available_recommendation_tables(
    catalog: Any,
    mapping: dict[str, str] | None = None,
    max_rows_per_model: int = 100_000,
) -> dict[str, pd.DataFrame]:
    """Load existing recommendation datasets as small Pandas tables."""
    mapping = mapping or MODEL_RECOMMENDATION_DATASETS
    tables = {}
    for model_name, dataset_name in mapping.items():
        data = _safe_load(catalog, dataset_name)
        if data is None:
            continue
        pdf = _to_pandas(data, max_rows_per_model)
        expected = {"user_idx", "item_idx", "score", "rank"}
        if expected.issubset(pdf.columns):
            tables[model_name] = pdf[list(expected)].copy()
    return tables


def load_ground_truth(
    catalog: Any,
    positive_threshold: float = 4.0,
    max_users: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a sampled test ground truth and corresponding user table."""
    test = _safe_load(catalog, "als_test")
    if test is None:
        return pd.DataFrame(columns=["user_idx", "item_idx"]), pd.DataFrame(
            columns=["user_idx"]
        )

    if hasattr(test, "filter"):
        from pyspark.sql import functions as F

        relevant = (
            test.filter(F.col("rating") >= positive_threshold)
            .select(F.col("user_idx").cast("int"), F.col("item_idx").cast("int"))
            .dropDuplicates(["user_idx", "item_idx"])
        )
        users = relevant.select("user_idx").distinct().orderBy(F.rand(seed))
        users = users.limit(max_users)
        relevant = relevant.join(users, "user_idx", "inner")
        return relevant.toPandas(), users.toPandas()

    pdf = _to_pandas(test)
    if "rating" in pdf.columns:
        pdf = pdf[pdf["rating"] >= positive_threshold]
    pdf = pdf[["user_idx", "item_idx"]].drop_duplicates()
    users = (
        pdf[["user_idx"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .head(max_users)
    )
    return pdf.merge(users, on="user_idx"), users


def load_dataset_overview(catalog: Any) -> dict[str, Any]:
    """Load small dataset-level statistics for the summary tab."""
    train = _safe_load(catalog, "als_train")
    test = _safe_load(catalog, "als_test")

    source = test if test is not None else train
    if source is None:
        return {
            "n_users": 0,
            "n_items": 0,
            "n_interactions": 0,
            "ratings": pd.DataFrame(columns=["rating", "count"]),
            "top_items": pd.DataFrame(columns=["item_idx", "interactions"]),
        }

    if hasattr(source, "select"):
        from pyspark.sql import functions as F

        n_users = source.select("user_idx").distinct().count()
        n_items = source.select("item_idx").distinct().count()
        n_interactions = source.count()
        ratings = (
            source.groupBy("rating")
            .agg(F.count("*").alias("count"))
            .orderBy("rating")
            .toPandas()
        )
        top_items = (
            source.groupBy("item_idx")
            .agg(F.count("*").alias("interactions"))
            .orderBy(F.col("interactions").desc())
            .limit(15)
            .toPandas()
        )
    else:
        pdf = _to_pandas(source)
        n_users = pdf["user_idx"].nunique() if "user_idx" in pdf else 0
        n_items = pdf["item_idx"].nunique() if "item_idx" in pdf else 0
        n_interactions = len(pdf)
        ratings = (
            pdf.groupby("rating", as_index=False).size().rename(columns={"size": "count"})
            if "rating" in pdf
            else pd.DataFrame(columns=["rating", "count"])
        )
        top_items = (
            pdf.groupby("item_idx", as_index=False)
            .size()
            .rename(columns={"size": "interactions"})
            .sort_values("interactions", ascending=False)
            .head(15)
            if "item_idx" in pdf
            else pd.DataFrame(columns=["item_idx", "interactions"])
        )

    sparsity = 1.0 - (n_interactions / (n_users * n_items)) if n_users and n_items else 0.0
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_interactions),
        "sparsity": float(sparsity),
        "ratings": ratings,
        "top_items": top_items,
    }


def filter_recommendations_to_users(
    recommendations_by_model: dict[str, pd.DataFrame],
    users: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Filter model recommendations to a sampled user set."""
    if users.empty:
        return recommendations_by_model
    user_ids = set(users["user_idx"])
    return {
        model: recs[recs["user_idx"].isin(user_ids)].copy()
        for model, recs in recommendations_by_model.items()
    }


def top_recommendations_with_hits(
    recommendations_by_model: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    user_idx: int,
    k: int,
) -> pd.DataFrame:
    """Return a side-by-side top-K table with hit flags for one user."""
    relevant = set(ground_truth.loc[ground_truth["user_idx"] == user_idx, "item_idx"])
    rows = []
    for model_name, recs in recommendations_by_model.items():
        user_recs = (
            recs[recs["user_idx"] == user_idx]
            .sort_values("rank")
            .head(k)
            .copy()
        )
        for row in user_recs.to_dict("records"):
            rows.append(
                {
                    "model": model_name,
                    "rank": int(row["rank"]),
                    "item_idx": int(row["item_idx"]),
                    "score": round(float(row.get("score", 0.0)), 6),
                    "hit": bool(row["item_idx"] in relevant),
                }
            )
    result = pd.DataFrame(rows)
    if "hit" in result:
        result["hit"] = result["hit"].astype(object)
    return result


def default_project_path() -> Path:
    """Return the current working directory as the default Kedro project root."""
    return Path.cwd()
