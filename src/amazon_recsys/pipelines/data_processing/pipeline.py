from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, index_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="amazon_reviews",           # Lee del catalog.yml (el parquet original)
                outputs="cleaned_reviews",         # Se guarda temporalmente en RAM
                name="clean_data_node",
            ),
            node(
                func=index_features,
                inputs="cleaned_reviews",          # Toma el dato limpio del paso anterior
                outputs="preprocessed_video_games",# Guarda el resultado final en catalog.yml
                name="index_features_node",
            ),
        ]
    )
