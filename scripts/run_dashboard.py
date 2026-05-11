"""Run the recommender evaluation Dash dashboard."""

from amazon_recsys.dashboard.app import create_app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=8050)
