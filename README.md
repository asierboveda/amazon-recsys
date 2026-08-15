# amazon-recsys

Recommender systems on the **Amazon Video Games (2023)** review dataset —
implemented as [Kedro](https://kedro.org) pipelines with PySpark and PyTorch,
evaluated with `recall@k` and explored through a Dash dashboard.

Three families of recommenders share one data-processing pipeline:

| Pipeline | Model | Stack |
|---|---|---|
| `recommender_als` | ALS (matrix factorization) | PySpark |
| `recommender_lightgcn` | LightGCN (graph convolutional) | PyTorch |
| `recommender_sasrec` | SASRec (self-attention, sequential) | PyTorch |

## What's here

- **Data processing** — raw `Video_Games.jsonl` → parquet, cleaning and
  train/test split ([scripts/download_data.py](scripts/download_data.py))
- **Three recommender pipelines** evaluated against a **popularity baseline**
  with a shared metric, `recall@k`, plus RMSE for ALS
  ([src/amazon_recsys/recommender_metrics.py](src/amazon_recsys/recommender_metrics.py))
- **Dash dashboard** — data overview, model metrics and per-user
  recommendations ([scripts/run_dashboard.py](scripts/run_dashboard.py))
- **Notebooks** — [data exploration](notebooks/01_data_exploration.ipynb) and
  [model evaluation](notebooks/02_model_evaluation.ipynb)
- **Docs** — algorithm notes and evaluation methodology
  ([docs/recommender_algorithms/](docs/recommender_algorithms/))
- **Tests** — pipeline, metrics and dashboard unit tests (`tests/`)

## Quickstart

```bash
pip install -r requirements.txt        # or: uv sync
python scripts/download_data.py        # Amazon Video Games 2023 → data/01_raw/
kedro run                              # run the full pipeline
pytest                                 # run the test suite
python scripts/run_dashboard.py        # open the dashboard
```

## Results

Evaluation runs produce `recall@k` per model, shown in the dashboard's model
tab. Result tables are not committed to the repository — the methodology is
described in [docs/recommender_algorithms/modelos_y_evaluacion_recsys.txt](docs/recommender_algorithms/modelos_y_evaluacion_recsys.txt).

## Status

Learning project focused on a solid engineering baseline: structured pipelines,
shared evaluation and a working dashboard. The three models are implemented and
unit-tested; training runs happen locally (the dataset is gitignored by Kedro
convention).

## Roadmap (ideas)

- Commit reproducible evaluation results (fixed seeds, `recall@k` tables)
- CI with pytest on GitHub Actions
- Live demo deployment of the dashboard

## License

MIT (proposed — pending confirmation).
