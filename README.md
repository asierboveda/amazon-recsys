# amazon-recsys

Proyecto Kedro para procesar reseñas de videojuegos de Amazon Reviews 2023 y comparar recomendaciones de popularidad, ALS, LightGCN y SASRec. Incluye un dashboard Dash y notebooks de exploración y evaluación. La descripción detallada de los datos, pipelines y métricas está en [context.md](context.md).

## Instalación

Se requiere Python 3.10 o superior y Java compatible con PySpark 3.5. Desde la raíz del repositorio, con [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Alternativamente, en un entorno virtual: `pip install -r requirements.txt` instala el proyecto y sus dependencias; `pip install -e '.[dev]'` añade las herramientas de desarrollo. Las dependencias se declaran en `pyproject.toml`; `uv.lock` fija la resolución para uv.

## Datos y ejecución

La descarga requiere acceso a Hugging Face y espacio suficiente para el dataset completo:

```bash
uv run python scripts/download_data.py
uv run kedro run
```

El comando predeterminado ejecuta el procesamiento de datos y ALS con el baseline de popularidad. Para ejecutar todos los modelos, incluyendo LightGCN y SASRec:

```bash
uv run kedro run --pipeline all_recommenders
```

También se pueden ejecutar los pipelines `data_processing`, `recommender_als`, `recommender_lightgcn` y `recommender_sasrec` por separado; los de modelos requieren las salidas de procesamiento. Los artefactos se guardan en las capas de `data/`, que no se versionan.

Para abrir el dashboard, después de generar los datos y recomendaciones disponibles:

```bash
uv run python scripts/run_dashboard.py
```

Visita http://127.0.0.1:8050. Los notebooks están en `notebooks/`; el dashboard y los notebooks pueden mostrar resultados parciales cuando faltan modelos.

## Desarrollo

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
# Ejecuta los cuatro métodos y el grafo completo con datos sintéticos aislados:
uv run python scripts/verify_synthetic_experiment.py
```

Las pruebas Spark necesitan Java. La configuración local y las credenciales pertenecen a `conf/local/` y no deben incluirse en Git.

## Interpretar el experimento

Los cuatro métodos usan interacciones positivas (rating ≥ 4) y se evalúan sobre los mismos usuarios y productos conocidos en entrenamiento. Las recomendaciones excluyen todos los productos vistos en entrenamiento y validación. Los modelos neuronales seleccionan el mejor checkpoint mediante Recall@20 en validación; test se reserva para el resultado final. El informe `data/08_reporting/experiment_stats.json` contabiliza los casos elegibles y las exclusiones.

La división es temporal por usuario y retrospectiva, no un corte global. El protocolo, los algoritmos y sus límites se explican en [context.md](context.md). Vuelve a ejecutar los pipelines después de actualizar el código: las recomendaciones y métricas generadas con versiones anteriores no son comparables con las nuevas.
