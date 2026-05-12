"""Pipeline definition for SASRec recommender."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_sasrec_sequences,
    evaluate_sasrec_ranking_metrics,
    generate_sasrec_recommendations,
    train_sasrec_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_sasrec_sequences,
                inputs=[
                    "als_train",
                    "als_validation",
                    "als_test",
                    "params:recommender_sasrec.data",
                ],
                outputs=[
                    "sasrec_train_sequences",
                    "sasrec_validation_sequences",
                    "sasrec_test_sequences",
                    "sasrec_seen_items",
                    "sasrec_dataset_stats",
                ],
                name="build_sasrec_sequences_node",
            ),
            node(
                func=train_sasrec_model,
                inputs=[
                    "sasrec_train_sequences",
                    "sasrec_validation_sequences",
                    "params:recommender_sasrec.model",
                ],
                outputs="sasrec_model",
                name="train_sasrec_model_node",
            ),
            node(
                func=generate_sasrec_recommendations,
                inputs=[
                    "sasrec_model",
                    "sasrec_test_sequences",
                    "sasrec_seen_items",
                    "params:recommender_sasrec.recommendations",
                ],
                outputs="sasrec_recommendations_top_k",
                name="generate_sasrec_recommendations_node",
            ),
            node(
                func=evaluate_sasrec_ranking_metrics,
                inputs=[
                    "sasrec_recommendations_top_k",
                    "sasrec_test_sequences",
                    "params:recommender_sasrec.evaluation",
                ],
                outputs="sasrec_ranking_metrics",
                name="evaluate_sasrec_ranking_metrics_node",
            ),
        ]
    )
