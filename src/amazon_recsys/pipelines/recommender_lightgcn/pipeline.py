"""Pipeline definition for LightGCN recommender."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_lightgcn_ranking_metrics,
    generate_lightgcn_recommendations,
    train_lightgcn_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_lightgcn_model,
                inputs=[
                    "positive_train",
                    "eligible_validation",
                    "params:recommender_lightgcn.model",
                    "als_train",
                ],
                outputs="lightgcn_model",
                name="train_lightgcn_model_node",
            ),
            node(
                func=generate_lightgcn_recommendations,
                inputs=[
                    "lightgcn_model",
                    "positive_train",
                    "eligible_test",
                    "params:recommender_lightgcn.recommendations",
                    "test_seen",
                ],
                outputs="lightgcn_recommendations_top_k",
                name="generate_lightgcn_recommendations_node",
            ),
            node(
                func=evaluate_lightgcn_ranking_metrics,
                inputs=[
                    "lightgcn_recommendations_top_k",
                    "eligible_test",
                    "params:recommender_lightgcn.evaluation",
                ],
                outputs="lightgcn_ranking_metrics",
                name="evaluate_lightgcn_ranking_metrics_node",
            ),
        ]
    )
