from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_popularity_baseline,
    compare_ranking_metrics,
    evaluate_popularity_baseline,
    evaluate_ranking_metrics,
    generate_als_recommendations,
    train_als_implicit_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_als_implicit_model,
                inputs=["positive_train", "params:recommender_als.als"],
                outputs="als_implicit_model",
                name="train_als_implicit_model_node",
            ),
            node(
                func=generate_als_recommendations,
                inputs=[
                    "als_implicit_model",
                    "positive_train",
                    "eligible_test",
                    "params:recommender_als.recommendations",
                    "test_seen",
                ],
                outputs="als_recommendations_top_k",
                name="generate_als_recommendations_node",
            ),
            node(
                func=evaluate_ranking_metrics,
                inputs=[
                    "als_recommendations_top_k",
                    "eligible_test",
                    "params:recommender_als.evaluation",
                ],
                outputs="als_ranking_metrics",
                name="evaluate_ranking_metrics_node",
            ),
            node(
                func=build_popularity_baseline,
                inputs=[
                    "positive_train",
                    "eligible_test",
                    "params:recommender_als.recommendations",
                    "test_seen",
                ],
                outputs="popularity_recommendations_top_k",
                name="build_popularity_baseline_node",
            ),
            node(
                func=evaluate_popularity_baseline,
                inputs=[
                    "popularity_recommendations_top_k",
                    "eligible_test",
                    "params:recommender_als.evaluation",
                ],
                outputs="popularity_ranking_metrics",
                name="evaluate_popularity_baseline_node",
            ),
            node(
                func=compare_ranking_metrics,
                inputs=[
                    "als_ranking_metrics",
                    "popularity_ranking_metrics",
                ],
                outputs="als_metrics_comparison",
                name="compare_model_metrics_node",
            ),
        ]
    )
