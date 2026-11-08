"""Project pipelines."""

from kedro.pipeline import Pipeline, node

from amazon_recsys.experiment import prepare_experiment

from amazon_recsys.pipelines.data_processing import (
    pipeline as data_processing_pipeline,
)
from amazon_recsys.pipelines.recommender_als import (
    pipeline as recommender_als_pipeline,
)
from amazon_recsys.pipelines.recommender_lightgcn import (
    pipeline as recommender_lightgcn_pipeline,
)
from amazon_recsys.pipelines.recommender_sasrec import (
    pipeline as recommender_sasrec_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing = data_processing_pipeline.create_pipeline()
    experiment = Pipeline(
        [
            node(
                prepare_experiment,
                ["als_train", "als_validation", "als_test", "params:experiment"],
                [
                    "positive_train",
                    "eligible_validation",
                    "eligible_test",
                    "test_seen",
                    "experiment_stats",
                ],
                name="prepare_experiment_node",
            )
        ]
    )
    recommender_als = recommender_als_pipeline.create_pipeline()
    recommender_lightgcn = recommender_lightgcn_pipeline.create_pipeline()
    recommender_sasrec = recommender_sasrec_pipeline.create_pipeline()

    return {
        "__default__": data_processing + experiment + recommender_als,
        "data_processing": data_processing,
        "recommender_als": experiment + recommender_als,
        "recommender_lightgcn": experiment + recommender_lightgcn,
        "recommender_sasrec": experiment + recommender_sasrec,
        "all_recommenders": data_processing
        + experiment
        + recommender_als
        + recommender_lightgcn
        + recommender_sasrec,
    }
