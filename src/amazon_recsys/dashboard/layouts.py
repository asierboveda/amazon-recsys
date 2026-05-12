"""Dash layouts for the recommender presentation dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html


CARD_STYLE = {
    "padding": "16px",
    "border": "1px solid #d7dde8",
    "borderRadius": "8px",
    "background": "#ffffff",
}


def metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
        className="metric-card",
    )


def build_layout(
    overview: dict,
    availability: pd.DataFrame,
    metrics_df: pd.DataFrame,
    user_options: list[dict],
) -> html.Div:
    """Build the full dashboard layout."""
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Amazon Recsys - Evaluacion de modelos"),
                    html.P(
                        "Comparacion de Popularity, ALS, LightGCN y SASRec sobre una muestra de usuarios de test."
                    ),
                ],
                className="header",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(label="Datos", children=data_tab(overview, availability)),
                    dcc.Tab(label="Modelos", children=models_tab(metrics_df)),
                    dcc.Tab(
                        label="Usuario",
                        children=user_tab(user_options),
                    ),
                ]
            ),
        ],
        className="page",
    )


def data_tab(overview: dict, availability: pd.DataFrame) -> html.Div:
    ratings = overview.get("ratings", pd.DataFrame())
    top_items = overview.get("top_items", pd.DataFrame())
    return html.Div(
        [
            html.Div(
                [
                    metric_card("Usuarios", f"{overview.get('n_users', 0):,}"),
                    metric_card("Items", f"{overview.get('n_items', 0):,}"),
                    metric_card(
                        "Interacciones", f"{overview.get('n_interactions', 0):,}"
                    ),
                    metric_card("Sparsity", f"{overview.get('sparsity', 0):.2%}"),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    dcc.Graph(
                        figure=px.bar(
                            ratings,
                            x="rating",
                            y="count",
                            title="Distribucion de ratings",
                        )
                    ),
                    dcc.Graph(
                        figure=px.bar(
                            top_items,
                            x="item_idx",
                            y="interactions",
                            title="Items mas frecuentes en la muestra base",
                        )
                    ),
                ],
                className="two-column",
            ),
            html.H3("Disponibilidad de modelos"),
            dash_table.DataTable(
                data=availability.to_dict("records"),
                columns=[{"name": col, "id": col} for col in availability.columns],
                page_size=8,
                style_table={"overflowX": "auto"},
            ),
        ],
        className="tab-body",
    )


def models_tab(metrics_df: pd.DataFrame) -> html.Div:
    k_options = sorted(metrics_df["k"].unique()) if not metrics_df.empty else [10, 20]
    return html.Div(
        [
            html.Div(
                [
                    html.Label("K"),
                    dcc.Dropdown(
                        id="k-selector",
                        options=[{"label": f"@{int(k)}", "value": int(k)} for k in k_options],
                        value=int(max(k_options)),
                        clearable=False,
                    ),
                ],
                className="control-row",
            ),
            dcc.Graph(id="metric-bars"),
            dash_table.DataTable(
                id="metrics-table",
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
            ),
        ],
        className="tab-body",
    )


def user_tab(user_options: list[dict]) -> html.Div:
    default_user = user_options[0]["value"] if user_options else None
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Usuario"),
                    dcc.Dropdown(
                        id="user-selector",
                        options=user_options,
                        value=default_user,
                        clearable=False,
                    ),
                ],
                className="control-row",
            ),
            html.Div(id="ground-truth-box", className="info-box"),
            dash_table.DataTable(
                id="user-recommendations-table",
                page_size=40,
                sort_action="native",
                style_table={"overflowX": "auto"},
            ),
        ],
        className="tab-body",
    )
