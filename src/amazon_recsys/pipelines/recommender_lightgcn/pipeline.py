"""Pipeline definition for LightGCN recommender."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_lightgcn_ranking_metrics,
    generate_lightgcn_recommendations,
    prepare_lightgcn_interactions,
    train_lightgcn_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_lightgcn_interactions,
                inputs=[
                    "als_train",
                    "als_validation",
                    "als_test",
                    "params:recommender_lightgcn.data",
                ],
                outputs=[
                    "lightgcn_train_interactions",
                    "lightgcn_validation_interactions",
                    "lightgcn_test_interactions",
                    "lightgcn_dataset_stats",
                ],
                name="prepare_lightgcn_interactions_node",
            ),
            node(
                func=train_lightgcn_model,
                inputs=[
                    "lightgcn_train_interactions",
                    "lightgcn_validation_interactions",
                    "params:recommender_lightgcn.model",
                ],
                outputs="lightgcn_model",
                name="train_lightgcn_model_node",
            ),
            node(
                func=generate_lightgcn_recommendations,
                inputs=[
                    "lightgcn_model",
                    "lightgcn_train_interactions",
                    "als_test",
                    "params:recommender_lightgcn.recommendations",
                ],
                outputs="lightgcn_recommendations_top_k",
                name="generate_lightgcn_recommendations_node",
            ),
            node(
                func=evaluate_lightgcn_ranking_metrics,
                inputs=[
                    "lightgcn_recommendations_top_k",
                    "lightgcn_test_interactions",
                    "params:recommender_lightgcn.evaluation",
                ],
                outputs="lightgcn_ranking_metrics",
                name="evaluate_lightgcn_ranking_metrics_node",
            ),
        ]
    )
