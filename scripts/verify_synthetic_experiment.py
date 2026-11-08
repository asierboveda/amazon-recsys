"""Run all Kedro nodes on an isolated, synthetic warm-start dataset.

No project data or checkpoints are overwritten.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from pyspark.sql import SparkSession
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from amazon_recsys.pipeline_registry import register_pipelines


class RetainedMemoryDataset(MemoryDataset):
    def _release(self):
        pass


def main():
    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    raw = spark.createDataFrame(
        [
            (f"u{u}", f"i{(u + j) % 6}", 5.0, j + 1, True, 0)
            for u in range(6)
            for j in range(4)
        ],
        "user_id string, parent_asin string, rating float, timestamp long, verified_purchase boolean, helpful_vote int",
    )
    with TemporaryDirectory() as d:
        params = {
            "params:data_processing.min_user_interactions": 3,
            "params:data_processing.min_item_interactions": 3,
            "params:data_processing.split": {
                "use_validation": True,
                "timestamp_col": "timestamp",
            },
            "params:experiment": {"positive_threshold": 4.0},
            "params:recommender_als.als": {
                "rank": 2,
                "maxIter": 1,
                "implicitPrefs": True,
                "model_path": str(Path(d) / "als"),
            },
            "params:recommender_als.recommendations": {"k": 3},
            "params:recommender_als.evaluation": {"k_values": [1, 3]},
            "params:recommender_lightgcn.model": {
                "embedding_dim": 4,
                "n_layers": 1,
                "epochs": 1,
                "batch_size": 4,
                "model_path": str(Path(d) / "lg.pt"),
            },
            "params:recommender_lightgcn.recommendations": {"k": 3},
            "params:recommender_lightgcn.evaluation": {"k_values": [1, 3]},
            "params:recommender_sasrec.data": {
                "max_seq_len": 4,
                "positive_threshold": 4.0,
            },
            "params:recommender_sasrec.model": {
                "hidden_size": 4,
                "inner_size": 8,
                "n_heads": 2,
                "n_layers": 1,
                "max_seq_len": 4,
                "epochs": 1,
                "batch_size": 4,
                "model_path": str(Path(d) / "sas.pt"),
            },
            "params:recommender_sasrec.recommendations": {"k": 3},
            "params:recommender_sasrec.evaluation": {"k_values": [1, 3]},
        }
        pipe = register_pipelines()["all_recommenders"]
        datasets = {
            name: RetainedMemoryDataset(copy_mode="assign") for name in pipe.datasets()
        }
        datasets["raw_video_games_reviews"] = MemoryDataset(
            data=raw, copy_mode="assign"
        )
        for name, val in params.items():
            datasets[name] = MemoryDataset(data=val, copy_mode="assign")
        catalog = DataCatalog(datasets=datasets)
        SequentialRunner().run(pipe, catalog)
        stats = catalog.load("experiment_stats")
        if stats["test"]["eligible"] != 6 or stats["validation"]["eligible"] != 6:
            raise AssertionError(f"Unexpected experiment population: {stats}")
        for model in ("als", "popularity", "lightgcn", "sasrec"):
            metrics = catalog.load(f"{model}_ranking_metrics")
            if metrics["recall@3"] != 1.0:
                raise AssertionError(f"Unexpected {model} metrics: {metrics}")
        for name in (
            "experiment_stats",
            "als_ranking_metrics",
            "popularity_ranking_metrics",
            "lightgcn_ranking_metrics",
            "sasrec_ranking_metrics",
        ):
            print(name, catalog.load(name))
    spark.stop()


if __name__ == "__main__":
    main()
