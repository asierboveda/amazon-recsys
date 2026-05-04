from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_popularity_baseline,
    compare_model_metrics,
    evaluate_popularity_baseline,
    evaluate_ranking_metrics,
    evaluate_rmse,
    generate_als_recommendations,
    train_als_explicit_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_als_explicit_model,
                inputs=["als_train", "params:recommender_als.als"],
                outputs="als_explicit_model",
                name="train_als_explicit_model_node",
            ),
            node(
                func=generate_als_recommendations,
                inputs=[
                    "als_explicit_model",
                    "als_test",
                    "params:recommender_als.recommendations",
                ],
                outputs="als_recommendations_top_k",
                name="generate_als_recommendations_node",
            ),
            node(
                func=evaluate_rmse,
                inputs=["als_explicit_model", "als_validation"],
                outputs="als_regression_metrics",
                name="evaluate_rmse_node",
            ),
            node(
                func=evaluate_ranking_metrics,
                inputs=[
                    "als_recommendations_top_k",
                    "als_test",
                    "params:recommender_als.evaluation",
                ],
                outputs="als_ranking_metrics",
                name="evaluate_ranking_metrics_node",
            ),
            node(
                func=build_popularity_baseline,
                inputs=[
                    "als_train",
                    "als_test",
                    "params:recommender_als.recommendations",
                ],
                outputs="popularity_recommendations_top_k",
                name="build_popularity_baseline_node",
            ),
            node(
                func=evaluate_popularity_baseline,
                inputs=[
                    "popularity_recommendations_top_k",
                    "als_test",
                    "params:recommender_als.evaluation",
                ],
                outputs="popularity_ranking_metrics",
                name="evaluate_popularity_baseline_node",
            ),
            node(
                func=compare_model_metrics,
                inputs=[
                    "als_regression_metrics",
                    "als_ranking_metrics",
                    "popularity_ranking_metrics",
                ],
                outputs="als_metrics_comparison",
                name="compare_model_metrics_node",
            ),
        ]
    )
