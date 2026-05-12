"""Dash app factory for the recommender dashboard."""

from __future__ import annotations

from pathlib import Path

from dash import Dash
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from .callbacks import register_callbacks
from .data_loader import (
    available_recommendation_tables,
    filter_recommendations_to_users,
    load_dataset_overview,
    load_ground_truth,
    model_availability,
)
from .layouts import build_layout
from .metrics import build_metric_table


def create_app(
    project_path: str | Path | None = None,
    max_test_users: int = 1000,
    positive_threshold: float = 4.0,
) -> Dash:
    """Create the Dash application using materialized Kedro outputs."""
    project_path = Path(project_path or Path.cwd())
    bootstrap_project(project_path)

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog

        availability = model_availability(catalog)
        overview = load_dataset_overview(catalog)
        ground_truth, users = load_ground_truth(
            catalog,
            positive_threshold=positive_threshold,
            max_users=max_test_users,
        )
        recommendations = available_recommendation_tables(catalog)
        recommendations = filter_recommendations_to_users(recommendations, users)

    catalog_size = int(overview.get("n_items", 0))
    metrics_df = build_metric_table(
        recommendations,
        ground_truth,
        catalog_size=catalog_size,
        k_values=[10, 20],
    )
    user_options = [
        {"label": str(int(user_idx)), "value": int(user_idx)}
        for user_idx in sorted(users["user_idx"].unique())
    ]

    app = Dash(__name__, title="Amazon Recsys Dashboard")
    app.layout = build_layout(overview, availability, metrics_df, user_options)
    app.index_string = INDEX_STRING
    register_callbacks(app, metrics_df, recommendations, ground_truth)
    return app


INDEX_STRING = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f4f6fa; color: #172033; font-family: Inter, Segoe UI, Arial, sans-serif; }
            .page { max-width: 1280px; margin: 0 auto; padding: 24px; }
            .header { margin-bottom: 18px; }
            .header h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
            .header p { margin: 0; color: #526070; }
            .tab-body { padding: 20px 0; }
            .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 14px; margin-bottom: 18px; }
            .metric-card { background: white; border: 1px solid #d7dde8; border-radius: 8px; padding: 16px; }
            .metric-label { color: #66758a; font-size: 13px; margin-bottom: 8px; }
            .metric-value { color: #172033; font-size: 24px; font-weight: 700; }
            .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
            .control-row { max-width: 260px; margin-bottom: 16px; }
            .control-row label { display: block; margin-bottom: 6px; font-weight: 600; }
            .info-box { background: white; border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
            @media (max-width: 900px) {
                .metric-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
                .two-column { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
