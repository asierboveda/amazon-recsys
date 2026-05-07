"""Nodes for the SASRec recommender pipeline."""

from __future__ import annotations

import random
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amazon_recsys.recommender_metrics import compute_recall_at_k

from .model import SASRecConfig, SASRecModel, require_torch


def _collect_user_sequences(df: DataFrame, item_id_offset: int) -> dict[int, list[int]]:
    rows = (
        df.select(
            F.col("user_idx").cast("int"),
            F.col("item_idx").cast("int"),
            F.col("timestamp"),
        )
        .orderBy("user_idx", "timestamp")
        .collect()
    )
    sequences: dict[int, list[int]] = {}
    for row in rows:
        sequences.setdefault(int(row["user_idx"]), []).append(
            int(row["item_idx"]) + item_id_offset
        )
    return sequences


def _truncate(sequence: list[int], max_seq_len: int) -> list[int]:
    return sequence[-max_seq_len:]


def _rows_to_df(spark: SparkSession, rows: list[tuple[int, list[int], int]]):
    schema = T.StructType(
        [
            T.StructField("user_idx", T.IntegerType(), False),
            T.StructField("sequence", T.ArrayType(T.IntegerType()), False),
            T.StructField("target_item", T.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)


def build_sasrec_sequences(
    als_train: DataFrame,
    als_validation: DataFrame,
    als_test: DataFrame,
    data_params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, dict]:
    """Build ordered per-user sequences for SASRec."""
    spark = SparkSession.builder.getOrCreate()
    max_seq_len = int(data_params.get("max_seq_len", 50))
    min_sequence_length = int(data_params.get("min_sequence_length", 1))
    item_id_offset = int(data_params.get("item_id_offset", 1))
    test_history_policy = data_params.get(
        "test_history_policy", "train_plus_validation"
    )

    train_by_user = _collect_user_sequences(als_train, item_id_offset)
    validation_by_user = _collect_user_sequences(als_validation, item_id_offset)
    test_by_user = _collect_user_sequences(als_test, item_id_offset)

    train_rows = []
    for user_idx, sequence in train_by_user.items():
        if len(sequence) >= min_sequence_length:
            train_rows.append(
                (user_idx, _truncate(sequence, max_seq_len), int(sequence[-1]))
            )

    validation_rows = []
    for user_idx, targets in validation_by_user.items():
        history = train_by_user.get(user_idx, [])
        if len(history) >= min_sequence_length:
            validation_rows.append(
                (user_idx, _truncate(history, max_seq_len), int(targets[0]))
            )

    test_rows = []
    for user_idx, targets in test_by_user.items():
        history = list(train_by_user.get(user_idx, []))
        if test_history_policy == "train_plus_validation":
            history.extend(validation_by_user.get(user_idx, []))
        if len(history) >= min_sequence_length:
            test_rows.append(
                (user_idx, _truncate(history, max_seq_len), int(targets[0]))
            )

    seen_source = als_train
    if test_history_policy == "train_plus_validation":
        seen_source = als_train.unionByName(als_validation)
    seen_items = seen_source.select(
        F.col("user_idx").cast("int"),
        F.col("item_idx").cast("int"),
    ).dropDuplicates(["user_idx", "item_idx"])

    all_item_max = []
    for df in [als_train, als_validation, als_test]:
        value = df.select(F.max("item_idx")).first()[0]
        if value is not None:
            all_item_max.append(int(value))

    stats = {
        "max_seq_len": max_seq_len,
        "min_sequence_length": min_sequence_length,
        "item_id_offset": item_id_offset,
        "test_history_policy": test_history_policy,
        "num_train_sequences": len(train_rows),
        "num_validation_sequences": len(validation_rows),
        "num_test_sequences": len(test_rows),
        "num_items": (max(all_item_max) + item_id_offset) if all_item_max else 0,
    }

    return (
        _rows_to_df(spark, train_rows),
        _rows_to_df(spark, validation_rows),
        _rows_to_df(spark, test_rows),
        seen_items,
        stats,
    )


def _pad_left(sequence: list[int], max_seq_len: int) -> list[int]:
    sequence = _truncate(sequence, max_seq_len)
    return [0] * (max_seq_len - len(sequence)) + sequence


def _training_examples(rows, max_seq_len: int) -> list[tuple[list[int], int]]:
    examples = []
    for row in rows:
        sequence = list(row["sequence"])
        for pos in range(1, len(sequence)):
            examples.append((_pad_left(sequence[:pos], max_seq_len), sequence[pos]))
    return examples


def _sample_negative(num_items: int, positive_items: set[int]) -> int:
    if len(positive_items) >= num_items:
        raise ValueError("Cannot sample a negative item when all items are positive.")
    while True:
        item_idx = random.randint(1, num_items)
        if item_idx not in positive_items:
            return item_idx


def train_sasrec_model(
    sasrec_train_sequences: DataFrame,
    sasrec_validation_sequences: DataFrame,
    model_params: dict,
) -> dict:
    """Train a SASRec model and persist the PyTorch checkpoint."""
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SASRec training on: {device}")

    seed = int(model_params.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    rows = sasrec_train_sequences.collect()
    max_seq_len = int(model_params.get("max_seq_len", 50))
    examples = _training_examples(rows, max_seq_len)
    if not examples:
        raise ValueError("Cannot train SASRec without at least one sequence example.")

    max_item_in_train = max(max(row["sequence"]) for row in rows if row["sequence"])
    num_items = int(model_params.get("num_items") or max_item_in_train)
    config = SASRecConfig(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=int(model_params.get("hidden_size", 64)),
        inner_size=int(model_params.get("inner_size", 256)),
        n_layers=int(model_params.get("n_layers", 2)),
        n_heads=int(model_params.get("n_heads", 2)),
        dropout=float(model_params.get("dropout", 0.2)),
    )
    model = SASRecModel(config)
    model.module.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_params.get("learning_rate", 0.001))
    )
    batch_size = int(model_params.get("batch_size", 256))
    epochs = int(model_params.get("epochs", 50))

    positives = {target for _, target in examples}
    last_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            
            # 1. Aseguramos que los 3 tensores de datos van a la gráfica
            sequences = torch.tensor([seq for seq, _ in batch], dtype=torch.long).to(device)
            pos_items = torch.tensor([target for _, target in batch], dtype=torch.long).to(device)
            neg_items = torch.tensor(
                [_sample_negative(num_items, positives) for _ in batch],
                dtype=torch.long,
            ).to(device)

            output = model.sequence_output(sequences)
            lengths = (sequences != 0).sum(dim=1).clamp(min=1) - 1
            
            # 2. LA TRAMPA: torch.arange crea tensores en CPU por defecto, le forzamos el device
            final = output[torch.arange(sequences.shape[0], device=device), lengths]
            
            item_weights = model.module.item_embedding.weight
            pos_scores = (final * item_weights[pos_items]).sum(dim=1)
            neg_scores = (final * item_weights[neg_items]).sum(dim=1)
            
            logits = torch.cat([pos_scores, neg_scores])
            labels = torch.cat(
                [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
            )
            
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    model_path = Path(model_params.get("model_path", "data/06_models/sasrec_model.pt"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config.__dict__, "state_dict": model.state_dict()},
        model_path,
    )

    return {
        "model_path": str(model_path),
        "num_items": num_items,
        "max_seq_len": max_seq_len,
        "hidden_size": config.hidden_size,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "last_loss": round(last_loss, 6),
        "num_validation_sequences": sasrec_validation_sequences.count(),
    }


def _load_sasrec(model_info: dict) -> SASRecModel:
    torch = require_torch()
    try:
        checkpoint = torch.load(
            model_info["model_path"], map_location="cpu", weights_only=True
        )
    except TypeError:  # pragma: no cover - older torch versions
        checkpoint = torch.load(model_info["model_path"], map_location="cpu")
    model = SASRecModel(SASRecConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def generate_sasrec_recommendations(
    sasrec_model: dict,
    sasrec_test_sequences: DataFrame,
    sasrec_seen_items: DataFrame,
    recommender_params: dict,
) -> DataFrame:
    """Generate top-K SASRec recommendations for test users."""
    torch = require_torch()
    spark = SparkSession.builder.getOrCreate()
    model = _load_sasrec(sasrec_model)

    k = int(recommender_params.get("k", 20))
    item_id_offset = int(recommender_params.get("item_id_offset", 1))
    max_seq_len = int(sasrec_model["max_seq_len"])

    seen_by_user: dict[int, set[int]] = {}
    for row in sasrec_seen_items.collect():
        seen_by_user.setdefault(int(row["user_idx"]), set()).add(int(row["item_idx"]))

    rows = sasrec_test_sequences.collect()
    output_rows = []
    with torch.no_grad():
        for row in rows:
            user_idx = int(row["user_idx"])
            sequence = _pad_left(list(row["sequence"]), max_seq_len)
            tensor = torch.tensor([sequence], dtype=torch.long)
            scores = model.scores(tensor)[0]
            scores[0] = -float("inf")
            for seen_item in seen_by_user.get(user_idx, set()):
                internal_item = seen_item + item_id_offset
                if internal_item < len(scores):
                    scores[internal_item] = -float("inf")
            limit = min(k, len(scores) - 1)
            values, indices = torch.topk(scores, k=limit)
            for rank, (score, internal_item) in enumerate(zip(values, indices), start=1):
                if torch.isinf(score):
                    continue
                item_idx = int(internal_item) - item_id_offset
                output_rows.append((user_idx, item_idx, float(score), rank))

    schema = T.StructType(
        [
            T.StructField("user_idx", T.IntegerType(), False),
            T.StructField("item_idx", T.IntegerType(), False),
            T.StructField("score", T.DoubleType(), False),
            T.StructField("rank", T.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(output_rows, schema=schema)


def evaluate_sasrec_ranking_metrics(
    sasrec_recommendations_top_k: DataFrame,
    sasrec_test_sequences: DataFrame,
    evaluation_params: dict,
) -> dict:
    """Compute Recall@K for SASRec recommendations."""
    item_id_offset = int(evaluation_params.get("item_id_offset", 1))
    ground_truth = sasrec_test_sequences.select(
        F.col("user_idx"),
        (F.col("target_item") - F.lit(item_id_offset)).cast("int").alias("item_idx"),
    )
    metrics = {}
    for k in evaluation_params.get("k_values", [10, 20]):
        recall = compute_recall_at_k(
            sasrec_recommendations_top_k, ground_truth, int(k)
        )
        metrics[f"recall@{k}"] = round(recall, 6)
    return metrics
