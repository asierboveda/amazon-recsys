"""Dash callback registration."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import Input, Output, html

from .data_loader import top_recommendations_with_hits


def register_callbacks(
    app,
    metrics_df: pd.DataFrame,
    recommendations_by_model: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    top_k: int = 20,
) -> None:
    """Register all dashboard callbacks."""

    @app.callback(
        Output("metric-bars", "figure"),
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Input("k-selector", "value"),
    )
    def update_model_metrics(k_value):
        subset = metrics_df[metrics_df["k"] == k_value].copy()
        if subset.empty:
            return px.bar(title="Sin metricas disponibles"), [], []

        long_df = subset.melt(
            id_vars=["model", "k"],
            value_vars=["recall", "precision", "ndcg", "coverage"],
            var_name="metric",
            value_name="value",
        )
        figure = px.bar(
            long_df,
            x="model",
            y="value",
            color="metric",
            barmode="group",
            title=f"Metricas de ranking @{k_value}",
        )
        table = subset.sort_values("recall", ascending=False)
        columns = [{"name": col, "id": col} for col in table.columns]
        return figure, table.to_dict("records"), columns

    @app.callback(
        Output("ground-truth-box", "children"),
        Output("user-recommendations-table", "data"),
        Output("user-recommendations-table", "columns"),
        Input("user-selector", "value"),
    )
    def update_user_recommendations(user_idx):
        if user_idx is None:
            return "No hay usuarios disponibles.", [], []

        truth_items = (
            ground_truth[ground_truth["user_idx"] == user_idx]["item_idx"]
            .astype(int)
            .tolist()
        )
        table = top_recommendations_with_hits(
            recommendations_by_model, ground_truth, int(user_idx), top_k
        )
        columns = [{"name": col, "id": col} for col in table.columns]
        return (
            html.Div(
                [
                    html.Strong(f"Ground truth test usuario {user_idx}: "),
                    html.Span(", ".join(map(str, truth_items)) or "sin items relevantes"),
                ]
            ),
            table.to_dict("records"),
            columns,
        )
