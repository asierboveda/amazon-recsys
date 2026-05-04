from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_recommender_interactions,
    deduplicate_user_item_interactions,
    filter_min_activity,
    index_user_item_ids,
    select_recommender_columns,
    temporal_train_validation_test_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_recommender_columns,
                inputs="raw_video_games_reviews",
                outputs="recsys_interactions_selected",
                name="select_recommender_columns_node",
            ),
            node(
                func=clean_recommender_interactions,
                inputs="recsys_interactions_selected",
                outputs="recsys_interactions_clean",
                name="clean_recommender_interactions_node",
            ),
            node(
                func=deduplicate_user_item_interactions,
                inputs="recsys_interactions_clean",
                outputs="recsys_interactions_deduplicated",
                name="deduplicate_user_item_interactions_node",
            ),
            node(
                func=filter_min_activity,
                inputs=[
                    "recsys_interactions_deduplicated",
                    "params:data_processing.min_user_interactions",
                    "params:data_processing.min_item_interactions",
                ],
                outputs="recsys_interactions_filtered",
                name="filter_min_activity_node",
            ),
            node(
                func=index_user_item_ids,
                inputs="recsys_interactions_filtered",
                outputs=[
                    "recsys_interactions_indexed",
                    "user_id_mapping",
                    "item_id_mapping",
                ],
                name="index_user_item_ids_node",
            ),
            node(
                func=temporal_train_validation_test_split,
                inputs=[
                    "recsys_interactions_indexed",
                    "params:data_processing.split",
                ],
                outputs=["als_train", "als_validation", "als_test"],
                name="temporal_train_validation_test_split_node",
            ),
        ]
    )
