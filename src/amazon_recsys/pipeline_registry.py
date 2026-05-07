"""Project pipelines."""

from kedro.pipeline import Pipeline

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
    recommender_als = recommender_als_pipeline.create_pipeline()
    recommender_lightgcn = recommender_lightgcn_pipeline.create_pipeline()
    recommender_sasrec = recommender_sasrec_pipeline.create_pipeline()

    return {
        "__default__": data_processing + recommender_als,
        "data_processing": data_processing,
        "recommender_als": recommender_als,
        "recommender_lightgcn": recommender_lightgcn,
        "recommender_sasrec": recommender_sasrec,
        "all_recommenders": data_processing
        + recommender_als
        + recommender_lightgcn
        + recommender_sasrec,
    }
