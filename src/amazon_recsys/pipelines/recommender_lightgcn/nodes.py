"""Nodes for the LightGCN recommender pipeline."""

from __future__ import annotations

import random
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amazon_recsys.recommender_metrics import compute_recall_at_k

from .model import (
    LightGCN,
    LightGCNConfig,
    bpr_loss,
    build_normalized_adj,
    require_torch,
)


def _positive_interactions(df: DataFrame, threshold: float) -> DataFrame:
    return (
        df.filter(F.col("rating") >= threshold)
        .select(
            F.col("user_idx").cast("int"),
            F.col("item_idx").cast("int"),
            F.col("rating").cast("float"),
        )
        .dropDuplicates(["user_idx", "item_idx"])
    )


def _max_index(dfs: list[DataFrame], column: str) -> int:
    max_values = []
    for df in dfs:
        value = df.select(F.max(column)).first()[0]
        if value is not None:
            max_values.append(int(value))
    return max(max_values) if max_values else -1


def prepare_lightgcn_interactions(
    train: DataFrame,
    validation: DataFrame,
    test: DataFrame,
    data_params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    """Prepare positive implicit-feedback interactions for LightGCN."""
    threshold = float(data_params.get("positive_threshold", 4.0))

    train_pos = _positive_interactions(train, threshold)
    validation_pos = _positive_interactions(validation, threshold)
    test_pos = _positive_interactions(test, threshold)

    num_users = _max_index([train, validation, test], "user_idx") + 1
    num_items = _max_index([train, validation, test], "item_idx") + 1

    stats = {
        "positive_threshold": threshold,
        "num_users": num_users,
        "num_items": num_items,
        "num_train_interactions": train_pos.count(),
        "num_validation_interactions": validation_pos.count(),
        "num_test_interactions": test_pos.count(),
    }
    return train_pos, validation_pos, test_pos, stats


def _collect_pairs(df: DataFrame) -> list[tuple[int, int]]:
    return [
        (int(row["user_idx"]), int(row["item_idx"]))
        for row in df.select("user_idx", "item_idx").collect()
    ]


def _sample_negative(
    num_items: int, positives_by_user: dict[int, set[int]], user_idx: int
) -> int:
    if len(positives_by_user[user_idx]) >= num_items:
        raise ValueError(
            f"User {user_idx} has interacted with every item; cannot sample negative."
        )
    while True:
        item_idx = random.randrange(num_items)
        if item_idx not in positives_by_user[user_idx]:
            return item_idx


def train_lightgcn_model(
    lightgcn_train_interactions: DataFrame,
    lightgcn_validation_interactions: DataFrame,
    model_params: dict,
) -> dict:
    """Train a LightGCN model and persist the PyTorch checkpoint."""
    torch = require_torch()

    seed = int(model_params.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    train_pairs = _collect_pairs(lightgcn_train_interactions)
    if not train_pairs:
        raise ValueError("Cannot train LightGCN without positive train interactions.")

    max_user = lightgcn_train_interactions.select(F.max("user_idx")).first()[0]
    max_item = lightgcn_train_interactions.select(F.max("item_idx")).first()[0]
    num_users = int(model_params.get("num_users") or int(max_user) + 1)
    num_items = int(model_params.get("num_items") or int(max_item) + 1)

    config = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=int(model_params.get("embedding_dim", 64)),
        n_layers=int(model_params.get("n_layers", 3)),
    )
    model = LightGCN(config)
    normalized_adj = build_normalized_adj(train_pairs, num_users, num_items)

    positives_by_user: dict[int, set[int]] = {}
    for user_idx, item_idx in train_pairs:
        positives_by_user.setdefault(user_idx, set()).add(item_idx)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_params.get("learning_rate", 0.001))
    )
    batch_size = int(model_params.get("batch_size", 2048))
    epochs = int(model_params.get("epochs", 50))
    reg_weight = float(model_params.get("reg_weight", 1e-4))

    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(train_pairs)
        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start : start + batch_size]
            users = [user_idx for user_idx, _ in batch]
            positives = [item_idx for _, item_idx in batch]
            negatives = [
                _sample_negative(num_items, positives_by_user, user_idx)
                for user_idx in users
            ]

            user_final, item_final = model.propagate(normalized_adj)
            user_tensor = torch.tensor(users, dtype=torch.long)
            pos_tensor = torch.tensor(positives, dtype=torch.long)
            neg_tensor = torch.tensor(negatives, dtype=torch.long)

            loss = bpr_loss(
                user_final[user_tensor],
                item_final[pos_tensor],
                item_final[neg_tensor],
                reg_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    model_path = Path(model_params.get("model_path", "data/06_models/lightgcn_model.pt"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "state_dict": model.state_dict(),
            "train_pairs": train_pairs,
        },
        model_path,
    )

    return {
        "model_path": str(model_path),
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": config.embedding_dim,
        "n_layers": config.n_layers,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "last_loss": round(last_loss, 6),
        "num_validation_interactions": lightgcn_validation_interactions.count(),
    }


def _load_lightgcn(model_info: dict):
    torch = require_torch()
    try:
        checkpoint = torch.load(
            model_info["model_path"], map_location="cpu", weights_only=True
        )
    except TypeError:  # pragma: no cover - older torch versions
        checkpoint = torch.load(model_info["model_path"], map_location="cpu")
    config = LightGCNConfig(**checkpoint["config"])
    model = LightGCN(config)
    model.load_state_dict(checkpoint["state_dict"])
    normalized_adj = build_normalized_adj(
        checkpoint["train_pairs"], config.num_users, config.num_items
    )
    return model, normalized_adj, config


def generate_lightgcn_recommendations(
    lightgcn_model: dict,
    lightgcn_train_interactions: DataFrame,
    als_test: DataFrame,
    recommender_params: dict,
) -> DataFrame:
    """Generate LightGCN top-K recommendations for users present in test."""
    torch = require_torch()
    spark = SparkSession.builder.getOrCreate()
    model, normalized_adj, config = _load_lightgcn(lightgcn_model)

    k = int(recommender_params.get("k", 20))
    score_batch_size = int(recommender_params.get("score_batch_size", 256))
    seen_by_user: dict[int, set[int]] = {}
    for user_idx, item_idx in _collect_pairs(lightgcn_train_interactions):
        seen_by_user.setdefault(user_idx, set()).add(item_idx)

    test_users = [
        int(row["user_idx"])
        for row in als_test.select("user_idx").distinct().collect()
        if int(row["user_idx"]) < config.num_users
    ]

    with torch.no_grad():
        user_final, item_final = model.propagate(normalized_adj)

    output_rows = []
    for start in range(0, len(test_users), score_batch_size):
        batch_users = test_users[start : start + score_batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long)
        scores = user_final[user_tensor] @ item_final.T

        for row_idx, user_idx in enumerate(batch_users):
            for seen_item in seen_by_user.get(user_idx, set()):
                if seen_item < config.num_items:
                    scores[row_idx, seen_item] = -float("inf")
            limit = min(k, config.num_items)
            values, indices = torch.topk(scores[row_idx], k=limit)
            for rank, (score, item_idx) in enumerate(zip(values, indices), start=1):
                if torch.isinf(score):
                    continue
                output_rows.append((user_idx, int(item_idx), float(score), rank))

    schema = T.StructType(
        [
            T.StructField("user_idx", T.IntegerType(), False),
            T.StructField("item_idx", T.IntegerType(), False),
            T.StructField("score", T.DoubleType(), False),
            T.StructField("rank", T.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(output_rows, schema=schema)


def evaluate_lightgcn_ranking_metrics(
    lightgcn_recommendations_top_k: DataFrame,
    lightgcn_test_interactions: DataFrame,
    evaluation_params: dict,
) -> dict:
    """Compute Recall@K for LightGCN recommendations."""
    metrics = {}
    for k in evaluation_params.get("k_values", [10, 20]):
        recall = compute_recall_at_k(
            lightgcn_recommendations_top_k,
            lightgcn_test_interactions,
            int(k),
        )
        metrics[f"recall@{k}"] = round(recall, 6)
    return metrics
